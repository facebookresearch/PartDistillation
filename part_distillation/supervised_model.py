# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import torch
import numpy as np
import detectron2.utils.comm as comm
import wandb

from torch import nn
from torch.nn import functional as F
from typing import Tuple, Union, Any, List
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .utils.utils import Partvisualizer, get_iou_all_cocoapi


@META_ARCH_REGISTRY.register()
class SupervisedModel(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        num_classes: int,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        test_topk_per_image: int,
        dataset_name: str="",
        # wandb
        use_wandb: bool=True,
        wandb_vis_period_train: int=200,
        wandb_vis_period_test: int=5,
        wandb_vis_topk: int=200,
        use_unique_per_pixel_label: bool=False,
        apply_masking_with_object_mask: bool=True,
        class_agnostic_learning: bool=True,
        class_agnostic_inference: bool=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.num_classes = num_classes
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.test_topk_per_image = test_topk_per_image
        self.cpu_device = torch.device("cpu")
        self.metadata = MetadataCatalog.get(dataset_name)
        self.logger = logging.getLogger("part_distillation")
        if self.metadata.get("part_classes") is None and self.metadata.get("thing_classes") is not None:
            self.metadata.part_classes = self.metadata.thing_classes

        # wandb
        self.use_wandb = use_wandb
        self.wandb_vis_period_train = wandb_vis_period_train
        self.wandb_vis_period_test = wandb_vis_period_test
        self.wandb_vis_topk = wandb_vis_topk
        self.num_train_iterations = 0
        self.num_test_iterations = 0

        self.use_unique_per_pixel_label = use_unique_per_pixel_label
        self.apply_masking_with_object_mask = apply_masking_with_object_mask
        self.class_agnostic_learning = class_agnostic_learning
        self.class_agnostic_inference = class_agnostic_inference


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # wandb
            "wandb_vis_period_train": cfg.WANDB.VIS_PERIOD_TRAIN,
            "wandb_vis_period_test": cfg.WANDB.VIS_PERIOD_TEST,
            "wandb_vis_topk": cfg.WANDB.VIS_TOPK,
            "use_wandb": not cfg.WANDB.DISABLE_WANDB,
            "dataset_name": cfg.DATASETS.TRAIN[0],
            "use_unique_per_pixel_label": cfg.SUPERVISED_MODEL.USE_PER_PIXEL_LABEL,
            "apply_masking_with_object_mask": cfg.SUPERVISED_MODEL.APPLY_MASKING_WITH_OBJECT_MASK,
            "class_agnostic_learning": cfg.SUPERVISED_MODEL.CLASS_AGNOSTIC_LEARNING,
            "class_agnostic_inference": cfg.SUPERVISED_MODEL.CLASS_AGNOSTIC_INFERENCE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }


    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # assert "instances" in batched_inputs[0], "gt should always be present. "
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        targets = self.prepare_targets(batched_inputs, images)
        outputs = self.sem_seg_head(features)

        if self.training:
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            if self.use_wandb and comm.is_main_process():
                if self.num_train_iterations % self.wandb_vis_period_train == 0:
                    processed_results_vis = self.inference(batched_inputs, targets, images, outputs, vis=True)
                    self.wandb_visualize(batched_inputs, images, processed_results_vis, is_training=True)
                    del processed_results_vis
            self.num_train_iterations += 1
            return losses
        else:
            processed_results = self.inference(batched_inputs, targets, images, outputs, vis=False)
            if self.use_wandb and comm.is_main_process():
                if self.num_test_iterations % self.wandb_vis_period_test == 0:
                    processed_results_vis = self.inference(batched_inputs, targets, images, outputs, vis=True)
                    self.wandb_visualize(batched_inputs, images, processed_results_vis, is_training=False)
                    del processed_results_vis
            self.num_test_iterations += 1

            return processed_results


    def inference(self, batched_inputs, targets, images, outputs, vis=False):
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        processed_results = []
        for batch_idx, (mask_cls_result, mask_pred_result, target, input_per_image, image_size) in enumerate(zip(
            mask_cls_results, mask_pred_results, targets, batched_inputs, images.image_sizes
        )):
            # NOTE: Unlike standard pipeline, we provide gt label as input for inference.
            #       This reshapes the labels to input size already, so we want to reshape
            #       both gts and predictions to the original image size.
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(mask_pred_result, image_size, height, width)
            target_masks = retry_if_cuda_oom(sem_seg_postprocess)(target["masks"].float(), image_size, height, width).bool()
            target_object_masks = retry_if_cuda_oom(sem_seg_postprocess)(target["object_masks"].float(), image_size, height, width).bool()
            mask_cls_result = mask_cls_result.to(mask_pred_result)

            processed_results.append({})
            if self.class_agnostic_learning or self.class_agnostic_inference:
                instance_r = self.instance_inference(mask_cls_result, mask_pred_result, target_masks, target_object_masks, \
                                                    target["labels"], vis=vis)
            else:
                instance_r = self.instance_inference_with_classification(mask_cls_result, mask_pred_result, target_masks, \
                                                    target_object_masks, target["labels"], vis=vis)
            target_inst = Instances(target_masks.shape[-2:])
            target_inst.gt_masks = target_masks
            target_inst.gt_classes = target["labels"]

            # For visualization
            target_inst.pred_masks = target_masks
            target_inst.pred_classes = target["labels"]

            processed_results[-1]["predictions"] = instance_r
            processed_results[-1]["gt_instances"] = target_inst
            processed_results[-1]["proposals"] = instance_r
            processed_results[-1]["gt_masks"] = target_inst

        return processed_results



    def _unique_assignment(self, masks_per_image, scores_per_image):
        obj_map_per_image = masks_per_image.topk(1, dim=0)[0] > 0.
        if self.use_unique_per_pixel_label:
            binmask_per_image  = masks_per_image > 0

            # predmask_per_image = masks_per_image.sigmoid()
            predmask_per_image = scores_per_image[:, None, None] * masks_per_image.sigmoid()

            scoremap_per_image = predmask_per_image.topk(1, dim=0)[1]
            query_indexs_list  = scoremap_per_image.unique()
            newmasks_per_image = masks_per_image.new_zeros(len(query_indexs_list), *scoremap_per_image.shape[1:])
            for i, cid in enumerate(query_indexs_list):
                newmasks_per_image[i] = (scoremap_per_image == cid) & obj_map_per_image
            scores_per_image = scores_per_image[query_indexs_list]

            return newmasks_per_image.bool(), scores_per_image

        else:
            return (masks_per_image > 0), scores_per_image




    def instance_inference_with_classification(self, mask_cls, mask_pred, target_masks, target_object_masks, target_labels, vis=False):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:] # QxHxW

        # [Q, K=1]
        wandb_vis_topk = self.wandb_vis_topk if vis and not self.use_unique_per_pixel_label else self.test_topk_per_image
        scores = mask_cls.softmax(-1)[:, :-1]
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(wandb_vis_topk, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.num_classes, rounding_mode='floor')
        mask_pred = mask_pred[topk_indices]  # KxHxW
        mask_pred = self.masking_with_object_mask(mask_pred, target_object_masks)

        # unique mapping and merging
        mask_pred_bool, scores_per_image, labels_per_image = self._unique_assignment_with_classes(mask_pred, scores_per_image, labels_per_image)

        mask_pred_bool, scores_per_image, labels_per_image, gt_part_labels = \
                    self.match_gt_labels_with_classification(mask_pred_bool, scores_per_image, labels_per_image, target_masks, target_labels)

        if mask_pred_bool.shape[0] == 0:
            # Doesn't contribute to the evaluation.
            mask_pred_bool = mask_pred.new_zeros(1, *mask_pred.shape[1:]).bool()
            scores_per_image = scores_per_image.new_zeros(1)
            labels_per_image = scores_per_image.new_ones(1).long() * self.num_classes
            gt_part_labels = scores_per_image.new_ones(1).long() * self.num_classes

        result = Instances(image_size)
        result.pred_masks = mask_pred_bool
        pred_masks_float = result.pred_masks.float()

        result.scores = scores_per_image
        result.pred_classes = labels_per_image

        return result



    def _unique_assignment_with_classes(self, masks_per_image, scores_per_image, class_labels):
        obj_map_per_image = masks_per_image.topk(1, dim=0)[0] > 0.
        if self.use_unique_per_pixel_label:
            # segmentation
            predmask_per_image = scores_per_image[:, None, None] * masks_per_image.sigmoid()
            labelmap_per_image = predmask_per_image.topk(1, dim=0)[1]
            query_indexs_list  = labelmap_per_image.unique()
            segmasks_per_image = masks_per_image.new_zeros(len(query_indexs_list), *labelmap_per_image.shape[1:])
            for i, cid in enumerate(query_indexs_list):
                segmasks_per_image[i] = (labelmap_per_image == cid) & obj_map_per_image
            scores_per_image = scores_per_image[query_indexs_list]
            class_labels = class_labels[query_indexs_list]

            # merging
            query_indexs_list  = class_labels.unique()
            newmasks_per_image = masks_per_image.new_zeros(len(query_indexs_list), *masks_per_image.shape[1:])
            newscore_per_image = scores_per_image.new_zeros(len(query_indexs_list))
            for i, cid in enumerate(query_indexs_list):
                newmasks_per_image[i] = segmasks_per_image[class_labels == cid].sum(dim=0).bool()
                newscore_per_image[i] = scores_per_image[class_labels == cid].topk(1, dim=0)[0].flatten()

            return newmasks_per_image.bool(), newscore_per_image, query_indexs_list
        else:
            return (masks_per_image > 0), scores_per_image, class_labels


    def prepare_targets(self, inputs, images):
        targets = [x["part_instances"].to(self.device) for x in inputs]
        object_targets = [x["instances"].to(self.device) for x in inputs]

        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for object_targets_per_image, targets_per_image in zip(object_targets, targets):
            gt_mask = targets_per_image.gt_masks.tensor
            padded_masks = torch.zeros((gt_mask.shape[0], h_pad, w_pad),
                                    dtype=gt_mask.dtype, device=gt_mask.device)
            padded_masks[:, : gt_mask.shape[1], : gt_mask.shape[2]] = gt_mask
            n = padded_masks.shape[0]

            gt_obj_mask = object_targets_per_image.gt_masks.tensor
            padded_obj_masks = torch.zeros((gt_obj_mask.shape[0], h_pad, w_pad),
                                    dtype=gt_obj_mask.dtype, device=gt_obj_mask.device)
            padded_obj_masks[:, : gt_obj_mask.shape[1], : gt_obj_mask.shape[2]] = gt_obj_mask

            labels = targets_per_image.gt_classes.to(self.device) \
                    if not self.class_agnostic_learning else torch.zeros(n).long().to(self.device)
            new_targets.append({"labels": labels,
                                "masks": padded_masks,
                                "object_masks": padded_obj_masks,
                                })

        return new_targets


    def masking_with_object_mask(self, masks_per_image, target_masks):
        if self.apply_masking_with_object_mask:
            object_target_mask = target_masks.sum(dim=0, keepdim=True).bool()

            return masks_per_image * object_target_mask
        else:
            return masks_per_image



    def instance_inference(self, mask_cls, mask_pred, target_masks, target_object_masks, target_labels, vis=False):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K=1]
        topk = self.wandb_vis_topk if vis and not self.use_unique_per_pixel_label else self.test_topk_per_image
        scores = mask_cls.softmax(-1)[:, :-1]

        if self.class_agnostic_learning:
            scores = scores.flatten() # Binary
        else:
            scores = scores.topk(1, dim=1)[0].flatten() # Use the top confidence score. (proposal eval only.)
        scores_per_image, topk_indices = scores.topk(topk, sorted=False)
        mask_pred = mask_pred[topk_indices]

        mask_pred = self.masking_with_object_mask(mask_pred, target_object_masks)
        mask_pred_bool, scores_per_image = self._unique_assignment(mask_pred, scores_per_image)

        mask_pred_bool, scores_per_image, gt_part_labels = \
                    self.match_gt_labels(mask_pred_bool, scores_per_image, target_masks, target_labels)

        if mask_pred_bool.shape[0] == 0:
            # Doesn't contribute to the evaluation.
            mask_pred_bool = mask_pred.new_zeros(1, *mask_pred.shape[1:]).bool()
            scores_per_image = scores_per_image.new_zeros(1)
            gt_part_labels = gt_part_labels.new_zeros(1)

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = mask_pred_bool
        pred_masks_float = result.pred_masks.float()

        # calculate average mask prob
        result.pred_classes = gt_part_labels
        result.scores = scores_per_image

        return result

    def register_metadata(self, dataset_name):
        self.logger.info("{} is registered for evaluation.".format(dataset_name))
        self.metadata = MetadataCatalog.get(dataset_name)
        if self.metadata.get("part_classes") is None and self.metadata.get("thing_classes") is not None:
            self.metadata.part_classes = self.metadata.thing_classes


    def match_gt_labels(self, masks_per_image, scores_per_image, target_masks, target_labels):
        pairwise_mask_ious = get_iou_all_cocoapi(masks_per_image, target_masks)

        top1_ious, top1_idx = pairwise_mask_ious.topk(1, dim=1)

        top1_idx = top1_idx.flatten()
        fg_idxs  = (top1_ious > 0.001).flatten()

        gt_part_labels = target_labels[top1_idx[fg_idxs]]
        masks_per_image = masks_per_image[fg_idxs]
        scores_per_image = scores_per_image[fg_idxs]

        return masks_per_image, scores_per_image, gt_part_labels


    def match_gt_labels_with_classification(self, masks_per_image, scores_per_image, labels_per_image, target_masks, target_labels):
        pairwise_mask_ious = get_iou_all_cocoapi(masks_per_image, target_masks)

        top1_ious, top1_idx = pairwise_mask_ious.topk(1, dim=1)

        top1_idx = top1_idx.flatten()
        fg_idxs  = (top1_ious > 0.001).flatten()

        gt_part_labels = target_labels[top1_idx[fg_idxs]]
        masks_per_image = masks_per_image[fg_idxs]
        scores_per_image = scores_per_image[fg_idxs]
        labels_per_image = labels_per_image[fg_idxs]

        return masks_per_image, scores_per_image, labels_per_image, gt_part_labels


    def wandb_visualize(self, inputs, images, processed_results, is_training, opacity=0.8):
        # NOTE: Hack to use input as visualization image.
        images_raw = [x["image"].float().to(self.cpu_device) for x in inputs]
        images_vis = [retry_if_cuda_oom(sem_seg_postprocess)(img, img_sz, x.get("height", img_sz[0]), x.get("width", img_sz[1]))
                        for img, img_sz, x in zip(images_raw, images.image_sizes, inputs)]
        images_vis = [img.to(self.cpu_device) for img in images_vis]
        result_vis = [r["predictions"].to(self.cpu_device) for r in processed_results]
        target_vis = [r["gt_instances"].to(self.cpu_device) for r in processed_results]
        image, instances, targets = images_vis[0], result_vis[0], target_vis[0]
        image = image.permute(1, 2, 0).to(torch.uint8)
        white = np.ones(image.shape) * 255
        image = image * opacity + white * (1-opacity)

        metadata = self.metadata if not is_training else None
        visualizer = Partvisualizer(image, metadata, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        image_pd = wandb.Image(vis_output.get_image())
        wandb.log({"predictions": image_pd})

        visualizer = Partvisualizer(image, metadata, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=targets)

        image_gt = wandb.Image(vis_output.get_image())
        wandb.log({"ground_truths": image_gt})
