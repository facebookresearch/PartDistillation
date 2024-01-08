# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import logging
import os 
import numpy as np
import torch
from typing import Tuple, Union, Any, List
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.data.transforms import TransformGen
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, BoxMode
import copy 

__all__ = ["ProposalDatasetMapper"]

# This is specifically designed for the COCO dataset.
class ProposalDatasetMapper:
    @configurable
    def __init__(
        self,
        image_format,
        base_aug,
        augs,
        weak_augs,
        instance_mask_format: str = "bitmask",
        min_object_area_ratio: float=0.0,
        min_area_ratio: float=0.0,
        class_code_to_class_id: dict={},
    ):
        self.base_aug = base_aug
        self.augs = augs
        self.weak_augs = weak_augs
        self.img_format = image_format
        self.instance_mask_format = instance_mask_format
        self.num_repeats = 100  # number of repeats until give up.
        self.min_object_area_ratio = min_object_area_ratio
        self.min_area_ratio = min_area_ratio
        self.class_code_to_class_id = class_code_to_class_id
        self.logger = logging.getLogger("part_distillation") 
    
    @classmethod
    def from_config(cls, cfg, is_train=True, base_size=-1):
        image_size = cfg.INPUT.IMAGE_SIZE
        aug_name_list = cfg.CUSTOM_DATASETS.AUG_NAME_LIST 
        class_code_to_class_id = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).class_code_to_class_id
        
        base_aug = []
        if base_size > 0:
            # Base size if pseudo-labels are done after resizing. 
            base_aug.append(T.ResizeScale(min_scale=1.0, 
                                          max_scale=1.0, 
                                          target_height=base_size, 
                                          target_width=base_size))

        augs, weak_augs = [], []
        if "flip" in aug_name_list:
            augs.append(T.RandomFlip())
            weak_augs.append(T.RandomFlip())
        if "color" in aug_name_list:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            weak_augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        if "rotation" in aug_name_list:
            augs.append(T.RandomRotation((0, 180)))
        if "crop" in aug_name_list:
            augs.append(T.RandomCrop(cfg.INPUT.CROP.TYPE,
                                     cfg.INPUT.CROP.SIZE,
                                     ))
        if "scale" in aug_name_list:
            min_scale = cfg.INPUT.MIN_SCALE
            max_scale = cfg.INPUT.MAX_SCALE
            augs.extend([T.ResizeScale(min_scale=min_scale, 
                                        max_scale=max_scale, 
                                        target_height=image_size, 
                                        target_width=image_size),
                         T.FixedSizeCrop(crop_size=(image_size, image_size)),
                        ])
        else:
            # No resizing but pad to make it square shape. 
            augs.extend([T.ResizeScale(min_scale=1.0, 
                                        max_scale=1.0, 
                                        target_height=image_size, 
                                        target_width=image_size),
                         T.FixedSizeCrop(crop_size=(image_size, image_size)),
                        ])
        weak_augs.extend([T.ResizeScale(min_scale=1.0, 
                                        max_scale=1.0, 
                                        target_height=image_size, 
                                        target_width=image_size),
                          T.FixedSizeCrop(crop_size=(image_size, image_size)),
                         ])
        ret = {
            "base_aug": base_aug,
            "augs": augs,
            "weak_augs": weak_augs,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "min_object_area_ratio": cfg.PROPOSAL_LEARNING.MIN_OBJECT_AREA_RATIO,
            "min_area_ratio": cfg.PROPOSAL_LEARNING.MIN_AREA_RATIO,
            "class_code_to_class_id": class_code_to_class_id,
        }

        return ret

    
    # If dataset is registered with [path_only] flag.
    def load_annotation(self, dataset_path, fname, ann_name):
        try:
            ann_dict = torch.load(os.path.join(dataset_path, fname, ann_name))
        except:
            self.logger.info(os.path.join(dataset_path, fname, ann_name), " is corrupted.")
            return 

        if ann_dict["object_ratio"] > self.min_object_area_ratio:
            new_dict = {"file_name": ann_dict["file_path"],
                        "image_id": ann_dict["file_name"],
                        "class_code": fname,
                        "height": None,
                        "width": None,
                        "pseudo_annotations": [], 
                        "gt_object_class": self.class_code_to_class_id[ann_dict["class_code"]],
                        }
            if ann_dict["part_mask"] is None or len(ann_dict["part_mask"]) == 0:
                return 
            for segm in ann_dict["part_mask"]:
                new_dict["pseudo_annotations"].append({"segmentation": segm["segmentation"],
                                                       "category_id": 0}) # class-agnostic -> postive = 0. 
                height, width = segm["segmentation"]["size"]
                new_dict["height"] = height 
                new_dict["width"]  = width

            if len(new_dict["pseudo_annotations"]) > 0:
                return new_dict
        


    def __call__(self, _dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        if isinstance(_dataset_dict, tuple):
            dataset_path, fname, ann_name = _dataset_dict
            _dataset_dict = self.load_annotation(dataset_path, fname, ann_name)
            if _dataset_dict is None:
                return 

        for _ in range(self.num_repeats):
            dataset_dict = self._forward(_dataset_dict, self.augs)
            if dataset_dict["instances"].has("gt_masks") \
            and len(dataset_dict["instances"]) > 0:  

                return dataset_dict 
        self.logger.info("Max number of repeats for data augmentation has reached.")
        self.logger.info("Processing with weak augmentation instead ...\n")
        dataset_dict = self._forward(_dataset_dict, self.weak_augs)
        assert dataset_dict["instances"].has("gt_masks")

        return dataset_dict



    def _forward(self, _dataset_dict, aug):
        dataset_dict = copy.deepcopy(_dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format) 
        if len(self.base_aug) > 0:
            image, _ = T.apply_transform_gens(self.base_aug, image)
        utils.check_image_size(dataset_dict, image)
        # print(image.shape, flush=True)
        padding_mask = np.zeros(image.shape[:2])
        image, transforms = T.apply_transform_gens(aug, image)
        padding_mask = transforms.apply_segmentation(padding_mask)
        # print(padding_mask.astype(bool).sum(), (~padding_mask.astype(bool)).sum(), flush=True)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape  = image.shape[:2]  # h, w
        # For visualization. 
        dataset_dict["height"] = image_shape[0]
        dataset_dict["width"] = image_shape[1]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        self._transform_annotations(dataset_dict, transforms, image_shape)
        del dataset_dict["pseudo_annotations"]

        return dataset_dict



    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        parts_list = dataset_dict["pseudo_annotations"]

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
        
        # NOTE: detectron2 pads 255 instead of 0 so make sure padding is correct. 
        if annos[0]['segmentation'].dtype == np.uint8:
            masks = torch.tensor([_['segmentation'] for _ in annos])
            for ann in annos:
                ann['segmentation'][ann['segmentation']==255] = 0

        # Convert to instances.
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        # print(640*640, instances.gt_masks.tensor.sum(), flush=True)
        if hasattr(instances, 'gt_masks'):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            instances = utils.filter_empty_instances(instances, by_box=False)

        new_instances = Instances(instances.image_size)
        masks = instances.gt_masks.tensor
        ratio = masks.flatten(1).sum(-1) / masks.sum()
        index = ratio > self.min_area_ratio
        new_instances.set("gt_masks", BitMasks(masks[index]))
        new_instances.set("gt_classes", torch.tensor(instances.gt_classes[index]))
        
        dataset_dict["instances"] = new_instances 

