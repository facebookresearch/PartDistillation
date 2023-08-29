# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import logging
import os 
import numpy as np
import torch
from typing import Tuple, Union, Any, List, Set
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, BoxMode


__all__ = ["PartDistillationDatasetMapper"]

# This is specifically designed for the COCO dataset.
class PartDistillationDatasetMapper:
    @configurable
    def __init__(
        self,
        image_format,
        base_aug,
        augs,
        weak_augs,
        instance_mask_format: str="bitmask", 
        min_object_area_ratio: float=-1.0,
        min_area_ratio: float=-1.0,
        min_score: float=-1.0, 
        class_code_to_class_id: dict={},
    ):
        self.base_aug = base_aug 
        self.augs = augs
        self.weak_augs = weak_augs
        self.img_format = image_format
        self.instance_mask_format = instance_mask_format
        self.num_repeats = 100  # number of repeats until give up.
        self.logger = logging.getLogger("part_distillation") 

        self.min_object_area_ratio = min_object_area_ratio
        self.min_area_ratio = min_area_ratio
        self.min_score = min_score 
        self.class_code_to_class_id = class_code_to_class_id
    

    @classmethod
    def from_config(cls, cfg, is_train=True):
        base_size  = cfg.CUSTOM_DATASETS.BASE_SIZE
        image_size = cfg.INPUT.IMAGE_SIZE
        aug_name_list = cfg.CUSTOM_DATASETS.AUG_NAME_LIST
        set_image_square = cfg.PART_DISTILLATION.SET_IMAGE_SQUARE
        
        # Need to resize to match GT. 
        base_aug = [T.ResizeScale(
                min_scale=1.0, max_scale=1.0, target_height=base_size, target_width=base_size
            ),]
        
        if set_image_square:
            # Fixing label bug from earlier ... 
            # some annotations are already in square format. 
            # TODO: remove when the bug is fixed. 
            base_aug.append(T.FixedSizeCrop(crop_size=(base_size, base_size)))

        augs, weak_augs = [], []
        if "flip" in aug_name_list:
            augs.append(T.RandomFlip())
            weak_augs.append(T.RandomFlip())
        if "color" in aug_name_list:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            weak_augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        if "rotation_45" in aug_name_list:
            augs.append(T.RandomRotation((0, 45))) 
        if "rotation_90" in aug_name_list:
            augs.append(T.RandomRotation((0, 90))) 
        if "rotation_180" in aug_name_list:
            augs.append(T.RandomRotation((0, 180))) 
        if "rotation" in aug_name_list:
            augs.append(T.RandomRotation((0, 360))) 
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
        test_aug = [] 

        class_code_to_class_id = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).class_code_to_class_id
        ret = {
            "base_aug": base_aug,
            "augs": augs if is_train else test_aug,
            "weak_augs": weak_augs if is_train else test_aug,
            "image_format": cfg.INPUT.FORMAT,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "class_code_to_class_id": class_code_to_class_id,
            "min_object_area_ratio": cfg.PART_DISTILLATION.MIN_OBJECT_AREA_RATIO, 
            "min_area_ratio": cfg.PART_DISTILLATION.MIN_AREA_RATIO, 
            "min_score": cfg.PART_DISTILLATION.MIN_SCORE,
        }

        return ret

    
    # If dataset is registered with [path_only] flag.
    def load_annotation(self, path_tuple):
        dataset_path, fname, ann_name = path_tuple
        try:   
            # print(0, dataset_path, flush=True)
            # NOTE: old annotations are saved before allocating to cpu, so map to cpu when loading.
            ann_dict = torch.load(os.path.join(dataset_path, fname, ann_name))
        except:
            self.logger.info("{} is corrupted.".format(os.path.join(dataset_path, fname, ann_name)))
            return  
        
        # filter object size 
        if ann_dict["object_ratio"] > self.min_object_area_ratio:
            new_dict = {"file_name": ann_dict["file_name"],
                        "image_id": ann_dict["image_id"],
                        "class_code": fname,
                        "height": None,
                        "width": None,
                        "pseudo_annotations": [], 
                        "gt_object_class": self.class_code_to_class_id[ann_dict["class_code"]],
                        }
            if ann_dict["part_masks"] is None or len(ann_dict["part_masks"]) == 0:
                return 

            for i, (lbl, segm) in enumerate(zip(ann_dict["part_labels"], ann_dict["part_masks"])):
                # filter each part size
                # if "part_ratios" not in ann_dict or ann_dict["part_ratios"][i] >= self.min_area_ratio:
                #     # filter each part score 
                #     if "part_scores" not in ann_dict or ann_dict["part_scores"][i] >= self.min_score:
                new_dict["pseudo_annotations"].append({"segmentation": segm["segmentation"],
                                                        "category_id": lbl}) 
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
            _dataset_dict = self.load_annotation(_dataset_dict)
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
        image_orig = utils.read_image(dataset_dict["file_name"], format=self.img_format) 
        image, _ = T.apply_transform_gens(self.base_aug, image_orig) 

        padding_mask = np.zeros(image.shape[:2])
        image, transforms = T.apply_transform_gens(aug, image)
        
        # the crop transformation has default padding value 0 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape  = image.shape[:2]  # h, w 
        
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"]  = image.shape[1]
        
        self._transform_annotations(dataset_dict, transforms, image_shape)

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
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


