# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import logging
import numpy as np
import detectron2.utils.comm as comm
import wandb

from torch import nn
from torch.nn import functional as F
from typing import Tuple
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.visualizer import ColorMode
from .utils.utils import proposals_to_coco_json, Partvisualizer, get_iou_all_cocoapi

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher



@META_ARCH_REGISTRY.register()
class PartDistillationModel(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        test_topk_per_image: int,
        # wandb
        use_wandb: bool=True,
        wandb_vis_period_train: int=200,
        wandb_vis_period_test: int=20,
        wandb_vis_topk: int=10,
        # postprocessing
        use_unique_per_pixel_label: bool=False,
        min_pseudo_mask_score: float=0.0,
        min_pseudo_mask_ratio: float=0.0,
        fg_score_threshold: float=0.1,
        train_dataset_name: str="",
        num_classes: int=8,
        use_oracle_classifier: bool=False,
        apply_masking_with_object_mask: bool=True,
    ):
        super().__init__()
        self.mode = ""

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.test_topk_per_image = test_topk_per_image
        self.fg_score_threshold = fg_score_threshold
        self.metadata = MetadataCatalog.get(train_dataset_name)
        self.train_dataset_name = train_dataset_name
        self.num_classes = num_classes

        # wandb
        self.use_wandb = use_wandb
        self.wandb_vis_period_train = wandb_vis_period_train
        self.wandb_vis_period_test = wandb_vis_period_test
        self.cpu_device = torch.device("cpu")
        self.wandb_vis_topk = wandb_vis_topk

        # postprocessing
        self.use_unique_per_pixel_label = use_unique_per_pixel_label
        self.min_pseudo_mask_score = min_pseudo_mask_score
        self.min_pseudo_mask_ratio = min_pseudo_mask_ratio
        self.majority_vote_mapping = {}
        self.current_train_iteration = 0
        self.current_test_iteration = 0
        self.use_oracle_classifier = use_oracle_classifier
        self.apply_masking_with_object_mask = apply_masking_with_object_mask

        # save parts
        self.root_save_path = "pseudo_labels/part_labels/part_distillation_predictions/{}/{}_{}/"\
                                .format(train_dataset_name, min_pseudo_mask_score, min_pseudo_mask_ratio)
        if comm.is_main_process():
            if not os.path.exists(self.root_save_path):
                os.makedirs(self.root_save_path)

            for fname in self.metadata.class_codes:
                folder_path = os.path.join(self.root_save_path, fname)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)



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
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_MATCH,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        num_classes = cfg.PART_DISTILLATION.NUM_PART_CLASSES
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_LOSS,
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
            # wandb
            "wandb_vis_period_train": cfg.WANDB.VIS_PERIOD_TRAIN,
            "wandb_vis_period_test": cfg.WANDB.VIS_PERIOD_TEST,
            "wandb_vis_topk": cfg.WANDB.VIS_TOPK,
            "use_wandb": not cfg.WANDB.DISABLE_WANDB,
            # postprocessing
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "use_unique_per_pixel_label": cfg.PART_DISTILLATION.USE_PER_PIXEL_LABEL,
            "train_dataset_name": cfg.DATASETS.TRAIN[0],
            "num_classes": num_classes,
            "min_pseudo_mask_ratio": cfg.PART_DISTILLATION.MIN_AREA_RATIO,
            "min_pseudo_mask_score": cfg.PART_DISTILLATION.MIN_SCORE,
            "use_oracle_classifier": cfg.PART_DISTILLATION.USE_ORACLE_CLASSIFIER,
            "apply_masking_with_object_mask": cfg.PART_DISTILLATION.APPLY_MASKING_WITH_OBJECT_MASK,
        }


    @property
    def device(self):
        return self.pixel_mean.device


    def register_metadata(self, dataset_name):
        self.logger.info("{} is registered for evaluation.".format(dataset_name))
        self.metadata = MetadataCatalog.get(dataset_name)


    def update_majority_vote_mapping(self, mapping_dict):
        self.logger.info("Updating class mapping based on majrotiy vote.")
        for cid, mapping in mapping_dict.items():
            self.majority_vote_mapping[cid] = mapping.to(self.device)


    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        targets = self.prepare_targets(batched_inputs, images)
        # NOTE: abusing the "mask" argument but works for now
        outputs = self.sem_seg_head(features, mask=targets)

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
                if self.current_train_iteration % self.wandb_vis_period_train == 0:
                    with torch.no_grad():
                        processed_results_vis = self.inference(batched_inputs, targets, images, outputs, vis=True)
                        self.wandb_visualize(batched_inputs, images, processed_results_vis)
                        del processed_results_vis

            self.current_train_iteration += 1
            return losses
        else:
            processed_results = self.inference(batched_inputs, targets, images, outputs, vis=False)
            if self.use_wandb and comm.is_main_process() and (self.mode == "eval" or self.mode == "save"):
                if self.current_test_iteration % self.wandb_vis_period_test == 0:
                    processed_results_vis = self.inference(batched_inputs, targets, images, outputs, vis=True)
                    self.wandb_visualize(batched_inputs, images, processed_results_vis)
                    del processed_results_vis

            self.current_test_iteration += 1
            return processed_results


    def inference(self, batched_inputs, targets, images, outputs, vis=False):
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"] # BxQxHxW

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
            target_mask = retry_if_cuda_oom(sem_seg_postprocess)(target["masks"].float(), image_size, height, width).bool()
            target_object_mask = retry_if_cuda_oom(sem_seg_postprocess)(target["object_mask"].float(), image_size, height, width).bool()
            mask_cls_result = mask_cls_result.to(mask_pred_result)

            processed_results.append({})

            instance_r = self.instance_inference_with_classification(mask_cls_result, mask_pred_result, target_mask, \
                                                    target_object_mask, target["labels"], target["gt_object_class"], vis=vis)
            if self.mode == "save" and not vis:
                self.save_part_segmentation(input_per_image, instance_r)

            processed_results[-1]["predictions"] = instance_r

            target_inst = Instances(target_mask.shape[-2:])
            target_inst.gt_masks = target_mask
            target_inst.gt_classes = target["labels"]

            # For visualization
            target_inst.pred_masks = target_mask
            target_inst.pred_classes = target["labels"]

            # For evaluation
            processed_results[-1]["gt_instances"] = target_inst
            processed_results[-1]["gt_object_label"] = target["gt_object_class"]

        return processed_results


    def save_part_segmentation(self, input_per_image, instance):
        if instance is not None:
            H, W = instance.pred_masks.shape[1:]
            object_area = instance.pred_masks.sum().long().item()
            part_areas = instance.pred_masks.flatten(1).sum(-1).long().cpu()
            image_area = H*W

            res = {"file_name": input_per_image["file_name"],
                    "image_id": input_per_image["image_id"],
                    "class_code": input_per_image["class_code"],
                    "height": H,
                    "width": W,
                    "part_masks": proposals_to_coco_json(instance.pred_masks.cpu()),
                    "part_labels": instance.pred_classes.cpu(),
                    "part_area_ratios":part_areas / object_area,
                    "object_ratio": object_area / image_area,
                    "part_scores": instance.scores.cpu().numpy()}
            torch.save(res, os.path.join(self.root_save_path, input_per_image["class_code"], input_per_image["image_id"]))

            del res
            del instance
            del input_per_image



    def masking_with_object_mask(self, masks_per_image, target_masks):
        if self.apply_masking_with_object_mask:
            object_target_mask = target_masks.sum(dim=0, keepdim=True).bool()

            return masks_per_image * object_target_mask
        else:
            return masks_per_image



    def match_gt_labels(self, masks_per_image, scores_per_image, prop_feats_per_image, target_masks, target_labels):
        # noqa
        pairwise_mask_ious = get_iou_all_cocoapi(masks_per_image, target_masks)

        top1_ious, top1_idx = pairwise_mask_ious.topk(1, dim=1)

        top1_idx = top1_idx.flatten()
        fg_idxs  = (top1_ious > self.fg_score_threshold).flatten()

        gt_part_labels = target_labels[top1_idx[fg_idxs]]
        masks_per_image = masks_per_image[fg_idxs]
        scores_per_image = scores_per_image[fg_idxs]
        prop_feats_per_image = prop_feats_per_image[fg_idxs]

        return masks_per_image, scores_per_image, prop_feats_per_image, gt_part_labels



    def _unique_assignment_with_classes(self, masks_per_image, scores_per_image, class_labels):
        obj_map_per_image = masks_per_image.topk(1, dim=0)[0] > 0.
        if self.use_unique_per_pixel_label:
            # segmentation
            predmask_per_image = scores_per_image[:, None, None] * masks_per_image.sigmoid()
            scoremap_per_image = predmask_per_image.topk(1, dim=0)[1]
            query_indexs_list  = scoremap_per_image.unique()
            segmasks_per_image = masks_per_image.new_zeros(len(query_indexs_list), *scoremap_per_image.shape[1:])
            for i, cid in enumerate(query_indexs_list):
                segmasks_per_image[i] = (scoremap_per_image == cid) & obj_map_per_image
            scores_per_image = scores_per_image[query_indexs_list]
            class_labels = class_labels[query_indexs_list]

            # merging
            new_class_labels = class_labels.unique()
            newmasks_per_image = masks_per_image.new_zeros(len(new_class_labels), *masks_per_image.shape[1:])
            newscore_per_image = scores_per_image.new_zeros(len(new_class_labels))
            for i, cid in enumerate(new_class_labels):
                newmasks_per_image[i] = segmasks_per_image[class_labels == cid].sum(dim=0).bool()
                newscore_per_image[i] = scores_per_image[class_labels == cid].topk(1, dim=0)[0].flatten()

            # filter
            loc_valid_idxs = newmasks_per_image.flatten(1).sum(dim=1) / obj_map_per_image.flatten(1).sum(dim=1) > self.min_pseudo_mask_ratio
            if loc_valid_idxs.any():
                newmasks_per_image = newmasks_per_image[loc_valid_idxs]
                newscore_per_image = newscore_per_image[loc_valid_idxs]
                new_class_labels = new_class_labels[loc_valid_idxs]

            loc_valid_idxs = newscore_per_image > self.min_pseudo_mask_score
            if loc_valid_idxs.any():
                newmasks_per_image = newmasks_per_image[loc_valid_idxs]
                newscore_per_image = newscore_per_image[loc_valid_idxs]
                new_class_labels = new_class_labels[loc_valid_idxs]

            return newmasks_per_image.bool(), newscore_per_image, new_class_labels
        else:
            # filter
            predmask_per_image = scores_per_image[:, None, None] * masks_per_image.sigmoid()
            loc_valid_idxs = (predmask_per_image > 0.5).flatten(1).sum(dim=1) / obj_map_per_image.flatten(1).sum(dim=1) > self.min_pseudo_mask_ratio
            if loc_valid_idxs.any():
                masks_per_image = predmask_per_image[loc_valid_idxs]
                scores_per_image = scores_per_image[loc_valid_idxs]
                class_labels = class_labels[loc_valid_idxs]

            loc_valid_idxs = scores_per_image > self.min_pseudo_mask_score
            if loc_valid_idxs.any():
                masks_per_image = masks_per_image[loc_valid_idxs]
                scores_per_image = scores_per_image[loc_valid_idxs]
                class_labels = class_labels[loc_valid_idxs]

            return (masks_per_image > 0), scores_per_image, class_labels



    def prepare_targets(self, inputs, images):
        if self.training or self.mode == "save":
            return self._prepare_pseudo_targets(inputs, images)
        else:
            return self._prepare_gt_targets(inputs, images)



    def _prepare_pseudo_targets(self, inputs, images):
        """
        return: Instance with gt_masks field.
        """
        pseudo_targets = [x["instances"].to(self.device) for x in inputs]
        h_pad, w_pad = images.tensor.shape[-2:] # NOTE: Assume same size for all images ?
        new_targets = []
        for idx, pseudo_targets_per_image in enumerate(pseudo_targets):
            gt_psuedo_masks = pseudo_targets_per_image.gt_masks.tensor
            padded_pseudo_masks = torch.zeros((gt_psuedo_masks.shape[0], h_pad, w_pad),
                                    dtype=gt_psuedo_masks.dtype, device=gt_psuedo_masks.device)
            padded_pseudo_masks[:, : gt_psuedo_masks.shape[1], : gt_psuedo_masks.shape[2]] = gt_psuedo_masks
            n = padded_pseudo_masks.shape[0]

            gt_labels = pseudo_targets_per_image.gt_classes.to(self.device)
            gt_object_class = inputs[idx]["gt_object_class"]

            new_targets.append({"labels": gt_labels,
                                "masks": padded_pseudo_masks,
                                "object_mask": padded_pseudo_masks.sum(dim=0, keepdim=True),
                                "gt_object_class": gt_object_class,
                                })

        return new_targets



    def _prepare_gt_targets(self, inputs, images):
        targets = [x["part_instances"].to(self.device) for x in inputs]
        object_targets = [x["instances"].to(self.device) for x in inputs]

        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for input_per_image, object_targets_per_image, targets_per_image in zip(inputs, object_targets, targets):
            gt_mask = targets_per_image.gt_masks.tensor
            padded_masks = torch.zeros((gt_mask.shape[0], h_pad, w_pad),
                                    dtype=gt_mask.dtype, device=gt_mask.device)
            padded_masks[:, : gt_mask.shape[1], : gt_mask.shape[2]] = gt_mask

            gt_obj_masks = object_targets_per_image.gt_masks.tensor
            padded_obj_masks = torch.zeros((gt_obj_masks.shape[0], h_pad, w_pad),
                                    dtype=gt_obj_masks.dtype, device=gt_obj_masks.device)
            padded_obj_masks[:, : gt_obj_masks.shape[1], : gt_obj_masks.shape[2]] = gt_obj_masks

            new_targets.append({"labels": targets_per_image.gt_classes.to(self.device),
                                "masks": padded_masks,
                                "object_mask": padded_obj_masks,
                                "gt_object_class": object_targets_per_image.gt_classes.to(self.device),
                                })

        return new_targets



    def instance_inference_with_classification(self, mask_cls, mask_pred, target_mask, target_object_mask, target_labels, target_object_label, vis=False):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        wandb_vis_topk = self.wandb_vis_topk if vis and not self.use_unique_per_pixel_label else self.test_topk_per_image
        scores = mask_cls.softmax(-1)[:, :-1]
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(wandb_vis_topk, sorted=False)
        labels_per_image = labels[topk_indices]

        if self.mode == "eval":
            majority_vote_mapping = self.majority_vote_mapping[target_object_label.item()]
            labels_per_image = majority_vote_mapping[labels_per_image]

        topk_indices = torch.div(topk_indices, self.num_classes, rounding_mode='floor')
        mask_pred = mask_pred[topk_indices]
        mask_pred = self.masking_with_object_mask(mask_pred, target_object_mask)

        # unique mapping and merging
        mask_pred_bool, scores_per_image, labels_per_image = self._unique_assignment_with_classes(mask_pred, scores_per_image, labels_per_image)
        mask_pred_bool, scores_per_image, labels_per_image, gt_part_labels = \
                    self.match_gt_labels(mask_pred_bool, scores_per_image, labels_per_image, target_mask, target_labels)

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

        if self.use_oracle_classifier:
            result.pred_classes = gt_part_labels
        else:
            result.pred_classes = labels_per_image

        return result




    def wandb_visualize(self, inputs, images, processed_results, opacity=0.8):
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

        visualizer = Partvisualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        image_pd = wandb.Image(vis_output.get_image())
        wandb.log({"predictions": image_pd})

        visualizer = Partvisualizer(image, self.metadata, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=targets)

        image_gt = wandb.Image(vis_output.get_image())
        wandb.log({"ground_truths": image_gt})
