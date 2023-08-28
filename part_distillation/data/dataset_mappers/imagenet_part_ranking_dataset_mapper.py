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
from detectron2.structures import BoxMode, Instances, BitMasks

__all__ = ["ImagenetPartRankingDatasetMapper"]

class ImagenetPartRankingDatasetMapper:
    @configurable
    def __init__(
        self,
        image_format,
        base_aug,
        aug,
        class_code_to_class_index,
        instance_mask_format: str = "bitmask",
    ):
        self.base_aug = base_aug
        self.aug = aug
        self.img_format = image_format
        self.class_code_to_class_index = class_code_to_class_index
        self.instance_mask_format = instance_mask_format

    @classmethod
    def from_config(cls, cfg, class_code_to_class_index):
        image_size = cfg.INPUT.IMAGE_SIZE

        # Need to resize to match GT.
        base_aug = [T.ResizeScale(
                    min_scale=1.0, max_scale=1.0, target_height=image_size, target_width=image_size),
                    ]
        aug = [T.FixedSizeCrop(crop_size=(image_size, image_size))]

        ret = {
            "base_aug": base_aug,
            "aug": aug,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "class_code_to_class_index": class_code_to_class_index,
        }

        return ret



    def __call__(self, dataset_dict):
        image_original = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image, _ = T.apply_transform_gens(self.base_aug, image_original)
        image_shape1 = image.shape

        image, transforms = T.apply_transform_gens(self.aug, image)
        image_shape = image.shape[:2]  # h, w

        dataset_dict["height"] = image_shape[0]
        dataset_dict["width"] = image_shape[1]
        self._transform_annotations(dataset_dict, transforms, image_shape)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        del dataset_dict["pseudo_annotations"]

        if len(dataset_dict["instances"]) > 0:
            return dataset_dict



    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        parts_list = dataset_dict["pseudo_annotations"]
        class_code = dataset_dict["class_code"]

        # NOTE: We do not use these information for pseudo label, but
        # to make the below functions happy we need them.
        # NOTE: set "by_box=False" for filtering empty instances !!!
        for part in parts_list:
            part["bbox"] = [0, 0, image_shape[0], image_shape[1]]
            part["bbox_mode"] = BoxMode.XYXY_ABS
            if "category_id" not in part:
                part["category_id"] = -1

        # Get flat list of annotations.
        annos = [utils.transform_instance_annotations(
                    part,
                    transforms,
                    image_shape,
                ) for part in parts_list]

        # Convert to instances.
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        if hasattr(instances, 'gt_masks'):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances, by_box=False)

        # Convert to object annotation by summing up all part masks.
        new_instances = Instances(instances.image_size)
        new_instances.set("gt_masks", BitMasks(instances.gt_masks.tensor.sum(0)[None]))
        new_instances.set("gt_classes", torch.tensor([self.class_code_to_class_index[class_code]]))
        dataset_dict["instances"] = new_instances
