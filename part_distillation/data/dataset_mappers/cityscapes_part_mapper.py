# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
import numpy as np
import torch
import pycocotools.mask as mask_util

from PIL import Image
from typing import Any, Dict, List, Set, Tuple
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, BoxMode
from detectron2.data.datasets.cityscapes import _cityscapes_files_to_dict, load_cityscapes_instances, _get_cityscapes_files
import panoptic_parts as pp


__all__ = ["CityscapesPartMapper"]


PART_CLASSES = (
        'person-torso', 'person-head', 'person-arm', 'person-leg',
        'rider-torso', 'rider-head', 'rider-arm', 'rider-leg',
        'car-window', 'car-wheel', 'car-light', 'car-license plate', 'car-chassis',
        'truck-window', 'truck-wheel', 'truck-light', 'truck-license plate', 'truck-chassis',
        'bus-window', 'bus-wheel', 'bus-light', 'bus-license plate', 'bus-chassis'
    )

PART_BASE_ID = {0: 0, 1: 4, 2: 8, 3: 13, 4: 18}
OBJECT_CLASSES = ('person', 'rider', 'car', 'truck', 'bus')



def load_object_and_parts(dict, file_path):
    """
    Object classes: 24, 25, 26, 27, 28 (5 classes).
        - Object class starts from 24 and ends with 28.
    Part classes:  15 + 8 = 23.
        - Part label starts from 1, and ends with either 4 or 5.
        - -1 is ignore, and 0 is unlabeled/void.
    """
    instances = utils.annotations_to_instances(dict["annotations"], (dict["height"], dict["width"]), mask_format="bitmask")
    if hasattr(instances, "gt_masks"):
        obj_classes = instances.gt_classes[instances.gt_classes < 5]
        obj_masks = instances.gt_masks.tensor[instances.gt_classes < 5].numpy()
        annos = [dict["annotations"][i] for i in range(len(instances.gt_classes)) if instances.gt_classes[i] < 5]

        img = np.array(Image.open(file_path))
        sids, iids, pids = pp.decode_uids(img)

        object_instances = []
        part_instances = []
        for instance_id, object_category_id in enumerate(obj_classes):
            object_category_id = object_category_id.item()
            object_dict = {"object_category": OBJECT_CLASSES[object_category_id],
                            "object_category_id": object_category_id,
                            "category_id": object_category_id, # For histogram printing.
                            "bbox": annos[instance_id]["bbox"],
                            "bbox_mode": annos[instance_id]["bbox_mode"],
                            "segmentation": mask_util.encode(np.asfortranarray(obj_masks[instance_id])),
                            }
            part_map = np.where(obj_masks[instance_id], pids, -1)

            part_instances_per_object = []
            for _pid in np.unique(part_map):
                # ignore -1 and 0.
                if _pid > 0:
                    part_id = PART_BASE_ID[object_category_id] + _pid-1
                    part_dict = {"part_category": PART_CLASSES[part_id],
                                "part_category_id": part_id, # shifting to make it 0 start.
                                "category_id": part_id, # For histogram printing.
                                "object_index": instance_id,
                                "segmentation": mask_util.encode(np.asfortranarray(np.where(part_map==_pid, True, False))),
                                }
                    part_instances_per_object.append(part_dict)
            object_instances.append(object_dict)
            part_instances.append(part_instances_per_object)

        return object_instances, part_instances
    else:
        return None, None



# This is specifically designed for the COCO dataset.
class CityscapesPartMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
        *
        tfm_gens,
        augmentations,
        aug_without_crop,
        image_format,
        size_divisibility,
        instance_mask_format: str = "bitmask",
        use_merged_gt: bool=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.is_train = is_train
        self.aug = augmentations
        self.aug_without_crop = aug_without_crop
        self.img_format = image_format
        self.size_divisibility = size_divisibility
        self.instance_mask_format = instance_mask_format
        self.num_repeats = 20  # number of repeats until give up.
        self.use_merged_gt = use_merged_gt

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        augs_without_crop = []
        if is_train:
            augs = [
                    T.ResizeShortestEdge(
                        cfg.INPUT.MIN_SIZE_TRAIN,
                        cfg.INPUT.MAX_SIZE_TRAIN,
                        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                    )
                ]
            augs.append(T.RandomFlip())
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs_without_crop = copy.deepcopy(augs)
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                    )
                )
        else:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TEST,
                    cfg.INPUT.MAX_SIZE_TEST,
                    "choice"
                )
            ]

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "aug_without_crop": augs_without_crop,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "use_merged_gt": cfg.CUSTOM_DATASETS.USE_MERGED_GT,
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

        self._transform_annotations(dataset_dict, transforms, image_shape)
        try:
            self._transform_part_annotations(dataset_dict, transforms, image_shape)
        except:
            return None
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        del dataset_dict["annotations"]
        del dataset_dict["part_annotations"]

        return dataset_dict



    def __call__(self, _dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if isinstance(_dataset_dict, tuple):
            dict, part_file = _dataset_dict

            object_instances, part_instances = load_object_and_parts(dict, part_file)
            if object_instances is not None:
                dict["annotations"] = object_instances
                dict["part_annotations"] = part_instances

                _dataset_dict = dict
            else:
                return None

        if self.is_train:
            for _ in range(self.num_repeats):
                dataset_dict = self._forward_with_aug(_dataset_dict, self.aug)
                if dataset_dict is not None \
                and "part_instances" in dataset_dict \
                and dataset_dict["part_instances"].has("gt_masks") \
                and len(dataset_dict["part_instances"]) > 0:
                    return dataset_dict

            return self._forward_with_aug(_dataset_dict, self.aug_without_crop)
        else:
            return self._forward_with_aug(_dataset_dict, self.aug)


    def _transform_annotations(self,
                               dataset_dict: Dict[str, Any],
                               transforms: Any,
                               image_shape: Tuple):
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape, keypoint_hflip_indices=False)
                    for obj in dataset_dict["annotations"]
                    if obj.get("iscrowd", 0) == 0
            ]
        instances = utils.annotations_to_instances(annos, image_shape,
                                    mask_format=self.instance_mask_format)
        obj_mapping = [obj_id for obj_id, obj in enumerate(dataset_dict["annotations"])]
        instances.obj_mapping = torch.tensor(obj_mapping, dtype=torch.int64)

        dataset_dict["instances"] = utils.filter_empty_instances(instances, by_box=False)


    def _transform_part_annotations(self,
                                    dataset_dict: Dict[str, Any],
                                    transforms: Any,
                                    image_shape: Tuple):
        parts_list = [
            part_ann
            for i, part_ann in enumerate(dataset_dict["part_annotations"])
            if i in dataset_dict["instances"].obj_mapping
        ]

        flat_part_segs = [
            part["segmentation"] for parts in list(parts_list) for part in parts
        ]

        for part_per_obj in parts_list:
            for part in part_per_obj:
                part["bbox"] = [0, 0, image_shape[0], image_shape[1]]
                part["bbox_mode"] = BoxMode.XYXY_ABS

        # The list of lists of parts will be flattened below, get mapping between a
        # part in the flat list and the object it corresponds to.
        obj_mapping = [obj_id for obj_id, obj in enumerate(parts_list) for _ in obj]

        # Get flat list of annotations.
        annos = [
            utils.transform_instance_annotations(
                part,
                transforms,
                image_shape,
                keypoint_hflip_indices=False,
            )
            for obj in parts_list
            for part in obj
        ]
        for _ann in annos:
            _ann["category_id"] = _ann["part_category_id"]

        # Convert to instances.
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        instances.obj_mapping = torch.tensor(obj_mapping, dtype=torch.int64)
        instances.part_mapping = torch.tensor(
            [i for i, _ in enumerate(flat_part_segs)], dtype=torch.int64
        )
        instances = utils.filter_empty_instances(instances, by_box=False)

        # save original part masks for evaluation
        dataset_dict["orig_part_maps"] = [
            parts
            for i, parts in enumerate(flat_part_segs)
            if i in instances.part_mapping
        ]

        if self.use_merged_gt:
            object_ids = instances.obj_mapping.unique()
            merged_msk = []
            gt_classes = []
            for oid in object_ids:
                part_mask_per_object = instances.gt_masks.tensor[instances.obj_mapping == oid]
                part_classes_per_object = instances.gt_classes[instances.obj_mapping == oid]
                for pid in part_classes_per_object.unique():
                    merged_msk.append(part_mask_per_object[part_classes_per_object==pid].sum(0).bool())
                    gt_classes.append(pid)
            new_instances = Instances(instances.image_size)
            new_instances.set("gt_masks", BitMasks(torch.stack(merged_msk, dim=0)))
            new_instances.set("gt_classes", torch.tensor(gt_classes))
        else:
            new_instances = instances

        dataset_dict["part_instances"] = new_instances
