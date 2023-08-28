# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import os
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from typing import Any, Dict, List, Set, Tuple
from pycocotools import mask as coco_mask


__all__ = ["PartImageNetMapper"]


def correct_part_imagenet_path(_dataset_dict):
    fname = _dataset_dict["file_name"].split('/')[-1].split('_')[0]
    iname = _dataset_dict["file_name"].split('/')[-1]
    path = "/".join(_dataset_dict["file_name"].split('/')[:-1])
    path = os.path.join(path, fname, iname)
    _dataset_dict["file_name"] = path
    _dataset_dict["class_code"] = fname



def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks



class PartImageNetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        *
        tfm_gens,
        augmentations,
        image_format,
        size_divisibility,
        use_merged_gt: bool=False,
        class_code_to_class_id: None,
    ):
        self.is_train = is_train
        self.aug = augmentations
        self.img_format = image_format
        self.size_divisibility = size_divisibility
        self.num_repeats = 20  # number of repeats until give up.
        self.use_merged_gt = use_merged_gt
        self.class_code_to_class_id = class_code_to_class_id


    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs = [
            T.ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]

        if is_train:
            augs.append(T.RandomFlip())
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                    )
                )

        # NOTE:This needs to be always from imagenet_1k_train!
        class_code_to_class_id = MetadataCatalog.get("imagenet_1k_meta_train").class_code_to_class_id

        # NOTE: Need to convert to the proper vocabulary.
        if "22k" in  cfg.DATASETS.TRAIN[0]:
            map_1k_to_22k = torch.load("metadata/imagenet1k_to_22k_mapping.pkl")
            class_code_to_class_id = {k: map_1k_to_22k[i] for k, i in class_code_to_class_id.items()}
        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "use_merged_gt": cfg.CUSTOM_DATASETS.USE_MERGED_GT,
            "class_code_to_class_id": class_code_to_class_id
        }

        return ret

    def _forward_with_aug(self, _dataset_dict, aug):
        dataset_dict = copy.deepcopy(_dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        aug_input, transforms = T.apply_transform_gens(aug, aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        dataset_dict["height"] = image_shape[0]
        dataset_dict["width"]  = image_shape[1]

        self._transform_part_annotations(dataset_dict, transforms, image_shape)
        self._transform_object_annotations(dataset_dict, transforms, image_shape)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if hasattr(dataset_dict["part_instances"], "gt_masks"):
            return dataset_dict



    def __call__(self, _dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        # part imagenet has incorrect path and need to be corrected.
        correct_part_imagenet_path(_dataset_dict)

        if self.is_train:
            for _ in range(self.num_repeats):
                dataset_dict = self._forward_with_aug(_dataset_dict, self.aug)
                if dataset_dict["part_instances"].has("gt_masks") \
                and len(dataset_dict["part_instances"]) > 0:
                    return dataset_dict

            return self._forward_with_aug(_dataset_dict, [])
        else:
            return self._forward_with_aug(_dataset_dict, self.aug)



    def _transform_part_annotations(self,
                               dataset_dict: Dict[str, Any],
                               transforms: Any,
                               image_shape: Tuple):
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=False)
                    for obj in dataset_dict["annotations"]
                    if obj.get("iscrowd", 0) == 0
            ]
        instances = utils.annotations_to_instances(annos, image_shape)
        h, w = instances.image_size
        if hasattr(instances, 'gt_masks'):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances)
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = BitMasks(gt_masks)
            if self.use_merged_gt:
                mask_all = instances.gt_masks.tensor
                label_all = instances.gt_classes
                unique_classes = instances.gt_classes.unique()
                merged_masks = []
                for c in unique_classes:
                    merged_masks.append(mask_all[label_all==c].sum(0))
                new_instances = Instances(instances.image_size)
                new_instances.set("gt_masks", BitMasks(torch.stack(merged_masks, dim=0)))
                new_instances.set("gt_classes", unique_classes)
                dataset_dict["part_instances"] = new_instances
            else:
                dataset_dict["part_instances"] = instances
        else:
            # some annotation has no part (will be resampled).
            dataset_dict["part_instances"] = instances

    def _transform_object_annotations(self,
                                      dataset_dict: Dict[str, Any],
                                      transforms: Any,
                                      image_shape: Tuple):

        if hasattr(dataset_dict["part_instances"], "gt_masks"):
            class_code = dataset_dict["class_code"]
            part_masks = dataset_dict["part_instances"].gt_masks.tensor
            new_instances = Instances(dataset_dict["part_instances"].image_size)
            new_instances.set("gt_masks", BitMasks(part_masks.sum(0, keepdim=True)))
            new_instances.set("gt_classes", torch.tensor([self.class_code_to_class_id[class_code]]))

            dataset_dict["instances"] = new_instances
