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
from detectron2.structures import BoxMode, Instances

__all__ = ["ProposalGenerationMapper"]


class ProposalGenerationMapper:
    @configurable
    def __init__(
        self,
        augmentations,
        image_format,
        with_given_mask: bool=False,
    ):
        self.aug = augmentations
        self.img_format = image_format
        self.with_given_mask = with_given_mask
        self.logger = logging.getLogger("part_distillation")

    @classmethod
    def from_config(cls, cfg):
        # Build augmentation
        image_size = cfg.INPUT.IMAGE_SIZE
        augs = [
            T.ResizeScale(
                min_scale=1.0, max_scale=1.0, target_height=image_size, target_width=image_size
            ),
            ]

        ret = {
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "with_given_mask": cfg.PROPOSAL_GENERATION.WITH_GIVEN_MASK
        }

        return ret

    def __call__(self, dataset_dict):
        try:
            image_original = utils.read_image(dataset_dict["file_path"], format=self.img_format)
        except:
            return
        utils.check_image_size(dataset_dict, image_original)

        image, _ = T.apply_transform_gens(self.aug, image_original)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.with_given_mask:
            self._transform_annotations(dataset_dict, [], image_shape)
            if not dataset_dict["instances"].has("gt_masks") or len(dataset_dict["instances"]) == 0:
                self.logger.info("No mask detected on {}.".format(dataset_dict["file_path"]))
                return None
            else:
                return dataset_dict
        else:
            return dataset_dict



    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        object_list = dataset_dict["pseudo_annotations"]

        # NOTE: We do not use these information for pseudo label, but
        # to make the below functions happy we need them.
        # NOTE: set "by_box=False" for filtering empty instances !!!
        for obj in object_list:
            obj["bbox"] = [0, 0, image_shape[0], image_shape[1]]
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            if "category_id" not in obj:
                obj["category_id"] = -1

        # Get flat list of annotations.
        annos = [utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                ) for obj in object_list]

        # Convert to instances.
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format="bitmask"
        )
        if hasattr(instances, 'gt_masks'):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances, by_box=False)

        dataset_dict["instances"] = instances
