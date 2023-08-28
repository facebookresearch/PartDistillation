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


@META_ARCH_REGISTRY.register()
class PartRankingModel(nn.Module):
    @configurable
    def __init__(
        self,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        num_queries: int,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        test_topk_per_image: int,
        # wandb
        use_wandb: bool=True,
        wandb_vis_period: int=10,
        wandb_vis_topk: int=10,
        # postprocess
        use_unique_per_pixel_label_during_labeling: bool=False,
        use_unique_per_pixel_label_during_clustering: bool=False,
        apply_masking_with_object_mask: bool=False,
        min_pseudo_mask_score_1: float=0.0,
        min_pseudo_mask_ratio_1: float=0.0,
        min_pseudo_mask_score_2: float=0.0,
        min_pseudo_mask_ratio_2: float=0.0,
        fg_score_threshold: float=0.001,
        num_clusters: int=8,
        proposal_key: str="decoder_output",
        classifier_metric: str="l2",
        dataset_name: str="",
        proposal_features_norm: bool=True,
        debug: bool=False,
    ):
        super().__init__()
        self.logger = logging.getLogger("part_distillation")
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.num_queries = num_queries
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # wandb
        self.use_wandb = use_wandb
        self.num_iters = 0
        self.wandb_vis_period = wandb_vis_period
        self.cpu_device = torch.device("cpu")
        self.wandb_vis_topk = wandb_vis_topk

        # postprocess
        self.mode = ""
        self.test_topk_per_image = test_topk_per_image
        self.proposal_features_norm = proposal_features_norm
        self.proposal_key = proposal_key
        self.classifier_metric = classifier_metric
        self.apply_masking_with_object_mask = apply_masking_with_object_mask
        self.use_unique_per_pixel_label_during_clustering = use_unique_per_pixel_label_during_clustering
        self.use_unique_per_pixel_label_during_labeling = use_unique_per_pixel_label_during_labeling
        self.min_pseudo_mask_score_1 = min_pseudo_mask_score_1
        self.min_pseudo_mask_ratio_1 = min_pseudo_mask_ratio_1
        self.min_pseudo_mask_score_2 = min_pseudo_mask_score_2
        self.min_pseudo_mask_ratio_2 = min_pseudo_mask_ratio_2
        self.fg_score_threshold = fg_score_threshold
        self.num_clusters = num_clusters
        self.classifier = {}
        self.majority_vote_mapping = {}

        # setup save dir
        dataset_name_dir = dataset_name.replace("_pre_labeling", "")if not debug else "debug"
        self.root_save_path = "pseudo_labels/part_labels/part_masks_with_class/{}/{}_{}/"\
                            .format(dataset_name_dir, classifier_metric, num_clusters)
        self.metadata = MetadataCatalog.get(dataset_name)
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

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # wandb
            "wandb_vis_period": cfg.WANDB.VIS_PERIOD_TEST,
            "wandb_vis_topk": cfg.WANDB.VIS_TOPK,
            "use_wandb": not cfg.WANDB.DISABLE_WANDB,
            # inference
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "apply_masking_with_object_mask": cfg.PART_RANKING.APPLY_MASKING_WITH_OBJECT_MASK,
            "use_unique_per_pixel_label_during_clustering": cfg.PART_RANKING.USE_PER_PIXEL_LABEL_DURING_CLUSTERING,
            "use_unique_per_pixel_label_during_labeling": cfg.PART_RANKING.USE_PER_PIXEL_LABEL_DURING_LABELING,
            "proposal_key": cfg.PART_RANKING.PROPOSAL_KEY,
            "classifier_metric": cfg.PART_RANKING.CLASSIFIER_METRIC,
            "dataset_name": cfg.DATASETS.TEST[0],
            "num_clusters": cfg.PART_RANKING.NUM_CLUSTERS,
            "proposal_features_norm": cfg.PART_RANKING.PROPOSAL_FEATURE_NORM,
            "min_pseudo_mask_ratio_1": cfg.PART_RANKING.MIN_AREA_RATIO_1,
            "min_pseudo_mask_score_1": cfg.PART_RANKING.MIN_SCORE_1,
            "min_pseudo_mask_ratio_2": cfg.PART_RANKING.MIN_AREA_RATIO_2,
            "min_pseudo_mask_score_2": cfg.PART_RANKING.MIN_SCORE_2,
            "debug": cfg.PART_RANKING.DEBUG,
        }


    def num_classes(self, k):
        return self.classifier[k.item()].weight.data.shape[0]


    def register_metadata(self, dataset_name):
        self.logger.info("{} is registered for evaluation.".format(dataset_name))
        self.metadata = MetadataCatalog.get(dataset_name)


    def update_majority_vote_mapping(self, mapping_dict):
        self.logger.info("Updating class mapping based on majrotiy vote.")
        for cid, mapping in mapping_dict.items():
            self.majority_vote_mapping[cid] = mapping.to(self.device)


    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs):
        assert not self.training, "part ranking is eval-only."
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)
        targets = self.prepare_targets(batched_inputs, images)

        processed_results = self.inference(batched_inputs, targets, images, outputs, vis=False)
        if self.use_wandb and comm.is_main_process():
            processed_results_vis = self.inference(batched_inputs, targets, images, outputs, vis=True)
            self.wandb_visualize(batched_inputs, images, processed_results_vis)
            del processed_results_vis

        return processed_results


    def inference(self, batched_inputs, targets, images, outputs, vis=False):
        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"] # BxQxHxW
        proposal_feats = outputs[self.proposal_key]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        if self.proposal_features_norm:
            proposal_feats = F.normalize(proposal_feats, p=2, dim=-1)

        processed_results = []
        for batch_idx, (mask_cls_result, mask_pred_result, proposal_feats_per_image, target, input_per_image, image_size) in enumerate(zip(
            mask_cls_results, mask_pred_results, proposal_feats, targets, batched_inputs, images.image_sizes
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

            if self.mode == "cluster":
                masks_per_image, scores_per_image, proposal_feats_per_image = \
                self.instance_inference_with_proposal_feats(proposal_feats_per_image,
                                                            mask_cls_result,
                                                            mask_pred_result,
                                                            target_mask,
                                                            target_object_mask,
                                                            vis=vis)
                masks_per_image, scores_per_image, proposal_feats_per_image = \
                self.match_gt_masks(masks_per_image,
                                    scores_per_image,
                                    proposal_feats_per_image,
                                    target_mask)

                result = Instances(image_size)
                result.pred_masks = masks_per_image.bool()
                result.scores = scores_per_image
                processed_results[-1]["predictions"] = result
                processed_results[-1]["proposal_features"] = proposal_feats_per_image
            else:
                instance_r = self.instance_inference_with_classification(proposal_feats_per_image,
                                                                         mask_cls_result,
                                                                         mask_pred_result,
                                                                         target_mask,
                                                                         target_object_mask,
                                                                         target["object_label"],
                                                                         vis=vis)
                processed_results[-1]["predictions"] = instance_r
                if not vis and self.mode == "save":
                    self.save_generated_part_labels(input_per_image, target["object_label"], instance_r)

            target_inst = Instances(target_mask.shape[-2:])
            target_inst.gt_masks = target_mask
            target_inst.pred_masks = target_mask    # for visualization
            if "part_labels" in target:
                target_inst.gt_classes = target["part_labels"]
            processed_results[-1]["gt_instances"] = target_inst
            processed_results[-1]["gt_object_label"] = target["object_label"]
            processed_results[-1]["gt_label"] = torch.tensor([target["object_label"] for _ in range(len(proposal_feats_per_image))])

        return processed_results



    def save_generated_part_labels(self, input_per_image, label, instance):
        if instance is not None:
            H, W = instance.pred_masks.shape[1:]
            res = {"file_name": input_per_image["file_name"],
                    "image_id": input_per_image["image_id"],
                    "class_code": input_per_image["class_code"],
                    "height": H,
                    "width": W,
                    "part_masks": proposals_to_coco_json(instance.pred_masks.cpu()),
                    "part_labels": instance.pred_classes.cpu(),
                    "object_ratio": instance.pred_masks.cpu().sum().long().item() / (H*W),
                    "part_ratios": instance.pred_masks.cpu().flatten(1).sum(-1) / (H*W),
                    "object_class_label": label.item(),
                    "part_scores": instance.scores.cpu().numpy()}
            torch.save(res, os.path.join(self.root_save_path, input_per_image["class_code"], input_per_image["image_id"]))



    def masking_with_object_mask(self, masks_per_image, target_masks):
        if self.apply_masking_with_object_mask:
            object_target_mask = target_masks.sum(dim=0, keepdim=True).bool()

            return masks_per_image * object_target_mask
        else:
            return masks_per_image



    def match_gt_masks(self, masks_per_image, scores_per_image, prop_feats_per_image, target_masks):
        pairwise_mask_ious = get_iou_all_cocoapi(masks_per_image, target_masks)

        top1_ious, top1_idx = pairwise_mask_ious.topk(1, dim=1)

        top1_idx = top1_idx.flatten()
        fg_idxs  = (top1_ious > self.fg_score_threshold).flatten()

        masks_per_image = masks_per_image[fg_idxs]
        scores_per_image = scores_per_image[fg_idxs]
        prop_feats_per_image = prop_feats_per_image[fg_idxs]

        return masks_per_image, scores_per_image, prop_feats_per_image



    def _unique_assignment_with_classes(self, masks_per_image, scores_per_image, class_labels):
        obj_map_per_image = masks_per_image.topk(1, dim=0)[0] > 0.
        if self.use_unique_per_pixel_label_during_labeling:
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
            loc_valid_idxs = newmasks_per_image.flatten(1).sum(dim=1) / obj_map_per_image.flatten(1).sum(dim=1) > self.min_pseudo_mask_ratio_2
            if loc_valid_idxs.any():
                newmasks_per_image = newmasks_per_image[loc_valid_idxs]
                newscore_per_image = newscore_per_image[loc_valid_idxs]
                new_class_labels = new_class_labels[loc_valid_idxs]

            loc_valid_idxs = newscore_per_image > self.min_pseudo_mask_score_2
            if loc_valid_idxs.any():
                newmasks_per_image = newmasks_per_image[loc_valid_idxs]
                newscore_per_image = newscore_per_image[loc_valid_idxs]
                new_class_labels = new_class_labels[loc_valid_idxs]

            return newmasks_per_image.bool(), newscore_per_image, new_class_labels
        else:
            # filter
            predmask_per_image = scores_per_image[:, None, None] * masks_per_image.sigmoid()
            loc_valid_idxs = (predmask_per_image > 0.5).flatten(1).sum(dim=1) / obj_map_per_image.flatten(1).sum(dim=1) > self.min_pseudo_mask_ratio_2
            if loc_valid_idxs.any():
                masks_per_image = predmask_per_image[loc_valid_idxs]
                scores_per_image = scores_per_image[loc_valid_idxs]
                class_labels = class_labels[loc_valid_idxs]

            loc_valid_idxs = scores_per_image > self.min_pseudo_mask_score_2
            if loc_valid_idxs.any():
                masks_per_image = masks_per_image[loc_valid_idxs]
                scores_per_image = scores_per_image[loc_valid_idxs]
                class_labels = class_labels[loc_valid_idxs]

            return (masks_per_image > 0), scores_per_image, class_labels



    def _unique_assignment(self, masks_per_image, scores_per_image, prop_feats_per_image, mask_prop_feats=None):
        obj_map_per_image = masks_per_image.topk(1, dim=0)[0] > 0.
        if self.use_unique_per_pixel_label_during_clustering:
            # unique assignment
            predmask_per_image = scores_per_image[:, None, None] * masks_per_image.sigmoid()
            scoremap_per_image = predmask_per_image.topk(1, dim=0)[1]
            query_indexs_list  = scoremap_per_image.unique()
            newmasks_per_image = masks_per_image.new_zeros(len(query_indexs_list), *scoremap_per_image.shape[1:])
            for i, cid in enumerate(query_indexs_list):
                newmasks_per_image[i] = (scoremap_per_image == cid) & obj_map_per_image
            scores_per_image = scores_per_image[query_indexs_list]
            prop_feats_per_image = prop_feats_per_image[query_indexs_list]

            # filter
            loc_valid_idxs = newmasks_per_image.flatten(1).sum(dim=1) / obj_map_per_image.flatten(1).sum(dim=1) > self.min_pseudo_mask_ratio_1
            if loc_valid_idxs.any():
                newmasks_per_image = newmasks_per_image[loc_valid_idxs]
                scores_per_image = scores_per_image[loc_valid_idxs]
                prop_feats_per_image = prop_feats_per_image[loc_valid_idxs]

            loc_valid_idxs = scores_per_image > self.min_pseudo_mask_score_1
            if loc_valid_idxs.any():
                newmasks_per_image = newmasks_per_image[loc_valid_idxs]
                scores_per_image = scores_per_image[loc_valid_idxs]
                prop_feats_per_image = prop_feats_per_image[loc_valid_idxs]

            return newmasks_per_image.bool(), scores_per_image, prop_feats_per_image
        else:
            # filter
            loc_valid_idxs = (masks_per_image > 0).flatten(1).sum(dim=1) / obj_map_per_image.flatten(1).sum(dim=1) > self.min_pseudo_mask_ratio_1
            if loc_valid_idxs.any():
                masks_per_image = masks_per_image[loc_valid_idxs]
                scores_per_image = scores_per_image[loc_valid_idxs]
                prop_feats_per_image = prop_feats_per_image[loc_valid_idxs]

            loc_valid_idxs = scores_per_image > self.min_pseudo_mask_score_1
            if loc_valid_idxs.any():
                masks_per_image = masks_per_image[loc_valid_idxs]
                scores_per_image = scores_per_image[loc_valid_idxs]
                prop_feats_per_image = prop_feats_per_image[loc_valid_idxs]

            return (masks_per_image > 0), scores_per_image, prop_feats_per_image


    def prepare_targets(self, inputs, images):
        if "part_instances" in inputs[0]:
            # evaluation
            part_targets = [x["part_instances"].to(self.device) for x in inputs]
            object_targets = [x["instances"].to(self.device) for x in inputs]
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for part_targets_per_image, object_targets_per_image in zip(part_targets, object_targets):
                gt_mask = part_targets_per_image.gt_masks.tensor
                padded_masks = torch.zeros((gt_mask.shape[0], h_pad, w_pad),
                                        dtype=gt_mask.dtype, device=gt_mask.device)
                padded_masks[:, : gt_mask.shape[1], : gt_mask.shape[2]] = gt_mask

                gt_obj_mask = object_targets_per_image.gt_masks.tensor
                padded_obj_mask = torch.zeros((gt_obj_mask.shape[0], h_pad, w_pad),
                                        dtype=gt_obj_mask.dtype, device=gt_obj_mask.device)
                padded_obj_mask[:, : gt_obj_mask.shape[1], : gt_obj_mask.shape[2]] = gt_obj_mask
                new_targets.append({"part_labels": part_targets_per_image.gt_classes.to(self.device),
                                    "object_label": object_targets_per_image.gt_classes.to(self.device),
                                    "masks": padded_masks,
                                    "object_mask": padded_obj_mask})
        else:
            #labeling
            targets = [x["instances"].to(self.device) for x in inputs]
            h_pad, w_pad = images.tensor.shape[-2:]
            new_targets = []
            for i, targets_per_image in enumerate(targets):
                gt_mask = targets_per_image.gt_masks.tensor
                padded_masks = torch.zeros((gt_mask.shape[0], h_pad, w_pad),
                                        dtype=gt_mask.dtype, device=gt_mask.device)
                padded_masks[:, : gt_mask.shape[1], : gt_mask.shape[2]] = gt_mask

                new_targets.append({"object_label": targets_per_image.gt_classes.to(self.device),
                                    "masks": padded_masks,
                                    "object_mask": padded_masks})

        return new_targets


    def register_classifier(self, centroids_dict):
        for cid, centroids in centroids_dict.items():
            num_cls, in_dim = centroids.shape
            self.classifier[cid] = nn.Linear(in_dim, num_cls, bias=False).to(self.device)
            self.classifier[cid].weight.data = centroids.to(self.device)


    def use_classifier(self, features, cid):
        if cid not in self.classifier:
            raise ValueError("class ID {} not in classifier. ({})".format(cid, self.classifier.keys()))

        if self.classifier_metric == "l2":
            # Efficient negative l2 distance implementation.
            y = self.classifier[cid].weight.data

            xy = self.classifier[cid](features)                  # NxK
            xx = (features * features).sum(dim=1)[:, None]       # Nx1
            yy = (y * y).sum(dim=1)                              # Kx1

            return xy - xx - yy.t()

        elif self.classifier_metric == "dot":
            return self.classifier[cid](features)


    def instance_inference_with_classification(self, proposal_feats, mask_cls, mask_pred, target_mask, target_object_mask, target_label, vis=False):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K=1]
        object_scores = mask_cls.softmax(-1)[:, :1]
        cls_outputs = self.use_classifier(proposal_feats, target_label.item())
        class_scores = cls_outputs.softmax(-1) # QxK

        # score = ranking score * confidence.
        scores = object_scores * class_scores
        topk = self.wandb_vis_topk if vis and not self.use_unique_per_pixel_label_during_labeling else self.test_topk_per_image
        labels = torch.arange(self.num_classes(target_label), device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten()
        scores_per_image, topk_indices = scores.flatten().topk(topk, sorted=False)

        if self.mode == "eval":
            labels_per_image = labels[topk_indices.flatten()].to(self.device)
            key = target_label.item()
            if len(self.majority_vote_mapping) != 0:
                majority_vote_mapping = self.majority_vote_mapping[key].to(self.device)
                labels_per_image = majority_vote_mapping[labels_per_image]
            else:
                raise ValueError("Class mapping is not registered.")
        else:
            labels_per_image = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.num_classes(target_label), rounding_mode='floor')
        mask_pred = mask_pred[topk_indices]

        # refine part mask with object mask
        mask_pred = self.masking_with_object_mask(mask_pred, target_object_mask)

        # unique mapping and merging
        mask_pred_bool, scores_per_image, labels_per_image = self._unique_assignment_with_classes(mask_pred, scores_per_image, labels_per_image)
        mask_pred_bool, scores_per_image, labels_per_image = \
                    self.match_gt_masks(mask_pred_bool, scores_per_image, labels_per_image, target_mask)

        if mask_pred_bool.shape[0] == 0:
            # doesn't contribute to the evaluation.
            mask_pred_bool = mask_pred.new_zeros(1, *mask_pred.shape[1:]).bool()
            scores_per_image = scores_per_image.new_zeros(1)
            labels_per_image = scores_per_image.new_zeros(1).long()

        result = Instances(image_size)
        result.pred_masks = mask_pred_bool
        pred_masks_float = result.pred_masks.float()
        result.scores = scores_per_image
        result.pred_classes = labels_per_image

        return result



    def instance_inference_with_proposal_feats(self, proposal_feats, mask_cls, mask_pred, target_mask, target_object_mask, vis=False):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K=1]
        scores = mask_cls.softmax(-1)[:, :-1].flatten()
        topk   = self.wandb_vis_topk if vis and not self.use_unique_per_pixel_label_during_clustering else self.test_topk_per_image
        scores_per_image, topk_indices = scores.topk(topk, sorted=False)
        masks_per_image = mask_pred[topk_indices]
        prop_feats_per_image = proposal_feats[topk_indices]

        # get unique assignment if needed
        masks_per_image, scores_per_image, prop_feats_per_image = self._unique_assignment(masks_per_image, scores_per_image, prop_feats_per_image)

        # refine part masks with object mask if needed
        masks_per_image = self.masking_with_object_mask(masks_per_image, target_object_mask)

        return masks_per_image, scores_per_image, prop_feats_per_image



    def wandb_visualize(self, inputs, images, processed_results, opacity=0.8):
        if self.num_iters % self.wandb_vis_period == 0:
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

        self.num_iters += 1
