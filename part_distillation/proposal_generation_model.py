# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
import detectron2.utils.comm as comm
import wandb

from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from typing import Tuple, List
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList, Instances
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.visualizer import ColorMode
from .utils.utils import proposals_to_coco_json, Partvisualizer



@META_ARCH_REGISTRY.register()
class ProposalGenerationModel(nn.Module):
    @configurable
    def __init__(
        self,
        backbone: Backbone,
        size_divisibility: int,
        dataset_name: str,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        distance_metric: str="l2",
        backbone_feature_key_list: List[str]=["res4"],
        num_superpixel_clusters: int=4,
        feature_normalize: bool=False,
        wandb_vis_period: int=100,
        debug: bool=False,
    ):
        super().__init__()
        self.backbone = backbone
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.cpu_device = torch.device("cpu")

        # clustering-related.
        self.distance_metric = distance_metric
        self.backbone_feature_key_list = backbone_feature_key_list
        self.num_superpixel_clusters = num_superpixel_clusters
        self.feature_normalize = feature_normalize
        self.kmeans_module = KMeans(n_clusters=num_superpixel_clusters, random_state=0)
        self.metadata = MetadataCatalog.get(dataset_name)
        self.num_test_iterations = 0
        self.wandb_vis_period = wandb_vis_period
        self.debug = debug

        self.root_save_path = self.metadata.save_path
        if comm.is_main_process():
            if not os.path.exists(self.root_save_path):
                os.makedirs(self.root_save_path)

            for fname in self.metadata.class_codes:
                folder_path = os.path.join(self.root_save_path, fname)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
        comm.synchronize()


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        return {
            "backbone": backbone,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY, # Set to 32.
            "dataset_name": cfg.PROPOSAL_GENERATION.DATASET_NAME,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "distance_metric": cfg.PROPOSAL_GENERATION.DISTANCE_METRIC,
            "backbone_feature_key_list": cfg.PROPOSAL_GENERATION.BACKBONE_FEATURE_KEY_LIST,
            "num_superpixel_clusters": cfg.PROPOSAL_GENERATION.NUM_SUPERPIXEL_CLUSTERS,
            "feature_normalize": cfg.PROPOSAL_GENERATION.FEATURE_NORMALIZE,
            "wandb_vis_period": cfg.WANDB.VIS_PERIOD_TEST,
            "debug": cfg.PROPOSAL_GENERATION.DEBUG,
        }

    @property
    def device(self):
        return self.pixel_mean.device


    def prepare_mask(self, inputs, images):
        # evaluation
        targets = [x["instances"].to(self.device) for x in inputs]
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_mask = targets_per_image.gt_masks.tensor
            padded_masks = torch.zeros((gt_mask.shape[0], h_pad, w_pad),
                                    dtype=gt_mask.dtype, device=gt_mask.device)
            padded_masks[:, : gt_mask.shape[1], : gt_mask.shape[2]] = gt_mask

            new_targets.append({"masks": padded_masks})

        return new_targets



    def _prepare_features(self, features):
        feat_dict = {key: features[key] for key in self.backbone_feature_key_list}
        H, W = features[self.backbone_feature_key_list[0]].shape[-2:]

        for k, v in feat_dict.items():
            feat_dict[k] = F.interpolate(v, size=(H, W), mode="bilinear", align_corners=False)

        feat_out = torch.cat([feat_dict[k] for k in self.backbone_feature_key_list], dim=1)
        if self.feature_normalize:
            feat_out = F.normalize(feat_out, dim=1, p=2)
        return feat_out



    def forward(self, batched_inputs):
        assert not self.training, "proposal generation is eval-only."
        with torch.no_grad():
            images = [x["image"].to(self.device) for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            targets = self.prepare_mask(batched_inputs, images)
            features = self.backbone(images.tensor)
            features = self._prepare_features(features)
            features_resized = F.interpolate(
                features,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            pseudo_label_list = []
            for input_per_image, feature_per_image, feature_resized_per_image, image_size, targets_per_image in \
                zip(batched_inputs, features, features_resized, images.image_sizes, targets):

                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                # Use masks for image-level scale (1 x) and feature-level scale (1/8 x)
                feature_resized_per_image = retry_if_cuda_oom(sem_seg_postprocess)(feature_resized_per_image, image_size, height, width)
                masks_resized = retry_if_cuda_oom(sem_seg_postprocess)(targets_per_image["masks"], image_size, height, width)[0].bool()
                masks = targets_per_image["masks"]
                masks = F.interpolate(masks[None].float(), size=feature_per_image.shape[-2:], mode="nearest")[0, 0].bool()
                pseudo_label = retry_if_cuda_oom(self.generate_pseudo_labels)(input_per_image, feature_per_image, feature_resized_per_image, masks, masks_resized)
                if pseudo_label is not None:
                    # save prediction
                    self.save_predictions(input_per_image, pseudo_label, masks_resized)

                    # For visualization
                    pseudo_label_list.append({})

                    instance = Instances(pseudo_label.shape[-2:])
                    instance.pred_masks = pseudo_label
                    instance.scores = pseudo_label.new_ones(pseudo_label.shape[0])

                    gt_instance = Instances(pseudo_label.shape[-2:])
                    gt_instance.pred_masks = pseudo_label

                    pseudo_label_list[-1]["proposals"] = instance
                    pseudo_label_list[-1]["gt_masks"] = gt_instance

            if comm.is_main_process() and self.num_test_iterations % self.wandb_vis_period == 0:
                if len(pseudo_label_list) > 0:
                    self.wandb_visualize(batched_inputs, images, pseudo_label_list)
            self.num_test_iterations += 1



    def save_predictions(self, input_per_image, pseudo_label, object_mask):
        H, W = object_mask.shape[-2:]

        res =  {"file_name": input_per_image["file_name"],
                "file_path": input_per_image["file_path"],
                "class_code": input_per_image["class_code"],
                "class_name": input_per_image["class_name"],
                "part_mask":  proposals_to_coco_json(pseudo_label),
                "object_ratio": object_mask.sum().long().item() / (H*W),
                "height": H,
                "width": W,
                "class_index": input_per_image["gt_object_class"],
                }

        torch.save(res, os.path.join(self.root_save_path, input_per_image["class_code"], input_per_image["file_name"]))


    def _get_superpixels(self, feature_per_image, pred_mask):
        H, W = feature_per_image.shape[-2:]
        data = feature_per_image[:, pred_mask].transpose(0, 1).contiguous().cpu()
        if len(data) > self.num_superpixel_clusters:
            kmeans  = self.kmeans_module.fit(data)
            centroids = kmeans.cluster_centers_
            centroids = torch.tensor(centroids).float()

            return centroids



    def _measure_distance(self, A, B):
        if self.distance_metric == "dot":
            return A @ B.T
        elif self.distance_metric == "l2":
            return  2 * A @ B.T - (A * A).sum(dim=1)[:, None] - (B * B).sum(1, keepdim=True).t()



    def generate_pseudo_labels(self, input_per_image, feature_per_image, feature_resized_per_image, object_mask, object_mask_resized):
        # clustering is done on small resolution (1/8).
        centroids = self._get_superpixels(feature_per_image, object_mask)
        if centroids is not None:
            feature_prop = feature_resized_per_image[:, object_mask_resized].transpose(0, 1).contiguous().cpu()
            pred_labels = self._measure_distance(feature_prop, centroids).topk(1, dim=1)[1].flatten() + 1

            mask = feature_prop.new_zeros(feature_resized_per_image.shape[-2:]).long()
            mask[torch.where(object_mask_resized==True)] = pred_labels

            pred_labels_unique = pred_labels.unique()
            binary_mask = mask.new_zeros(len(pred_labels_unique), *feature_resized_per_image.shape[-2:]).bool() # PxHxW
            for i, plbl in enumerate(pred_labels_unique):
                binary_mask[i] = mask == plbl

            return binary_mask



    def wandb_visualize(self, inputs, images, processed_results, opacity=0.8):
        # NOTE: Hack to use input as visualization image.
        images_raw = [x["image"].float().to(self.cpu_device) for x in inputs]
        images_vis = [retry_if_cuda_oom(sem_seg_postprocess)(img, img_sz, x.get("height", img_sz[0]), x.get("width", img_sz[1]))
                        for img, img_sz, x in zip(images_raw, images.image_sizes, inputs)]
        images_vis = [img.to(self.cpu_device) for img in images_vis]
        result_vis = [r["proposals"].to(self.cpu_device) for r in processed_results]
        target_vis = [r["gt_masks"].to(self.cpu_device) for r in processed_results]
        image, instances, targets = images_vis[0], result_vis[0], target_vis[0]
        image = image.permute(1, 2, 0).to(torch.uint8)
        white = np.ones(image.shape) * 255
        image = image * opacity + white * (1-opacity)

        targets = Instances(instances.pred_masks.shape[-2:])
        visualizer = Partvisualizer(image, None, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=targets)

        image_gt = wandb.Image(vis_output.get_image())
        wandb.log({"ground_truths": image_gt})

        visualizer = Partvisualizer(image, None, instance_mode=ColorMode.IMAGE)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        image_pd = wandb.Image(vis_output.get_image())
        wandb.log({"predictions": image_pd})
