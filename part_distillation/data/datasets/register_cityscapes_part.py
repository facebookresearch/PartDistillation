# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import copy 
import numpy as np
import pycocotools.mask as mask_util
import panoptic_parts as pp

from typing import Tuple
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes import load_cityscapes_instances
from detectron2.data import detection_utils as utils
from detectron2.utils.file_io import PathManager
from PIL import Image 

CITYSCAPES_DATASET_ROOT = "datasets/cityscapes_part/"
CITYSCAPES_DATASET_IMAGES = CITYSCAPES_DATASET_ROOT + "leftImg8bit/"
CITYSCAPES_DATASET_PART_ANNS = CITYSCAPES_DATASET_ROOT + "gtFinePanopticParts/"
CITYSCAPES_DATASET_OBJ_ANNS = CITYSCAPES_DATASET_ROOT + "gtFine/"

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
    anns_size = (dict["height"], dict["width"])
    instances = utils.annotations_to_instances(dict["annotations"], 
                                               anns_size, 
                                               mask_format="bitmask")
    
    object_instances = []
    part_instances = [] 
    if hasattr(instances, "gt_masks"):
        obj_classes = instances.gt_classes[instances.gt_classes < 5]
        obj_masks = instances.gt_masks.tensor[instances.gt_classes < 5].numpy()
        annos = [dict["annotations"][i] for i in range(len(instances.gt_classes)) if instances.gt_classes[i] < 5]

        img = np.array(Image.open(file_path))
        sids, iids, pids = pp.decode_uids(img)
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
                    part_id = PART_BASE_ID[object_category_id] + _pid-1 # shifting to make it 0 start. 
                    part_dict = {"part_category": PART_CLASSES[part_id],
                                "part_category_id": part_id, 
                                "category_id": part_id,      # For histogram printing.
                                "object_index": instance_id,
                                "segmentation": mask_util.encode(np.asfortranarray(np.where(part_map==_pid, True, False))),
                                }
                    part_instances_per_object.append(part_dict)
            
            # some object has no parts.
            if len(part_dict) > 0:
                object_instances.append(object_dict)
                part_instances.append(part_instances_per_object)
        
        return object_instances, part_instances



def load_cityscapes_object_part_instances(
    images_dirname: str,
    annotations_dirname: str,
    split: str,
    path_only: bool=False,
    label_percentage: int=100, 
    for_segmentation: bool=False, 
    debug: bool=False,
):  
    logger = logging.getLogger("part_distillation")
    logger.info("Starting loading cityscapes part data")
    
    if len(images_dirname) == 0:
        images_dirname = CITYSCAPES_DATASET_IMAGES 
    if len(annotations_dirname) == 0:
        annotations_dirname = CITYSCAPES_DATASET_OBJ_ANNS 
    original_dicts = load_cityscapes_instances(images_dirname + split, annotations_dirname + split)

    if label_percentage < 100:
        # shuffle and pick the first n.
        np.random.seed(1234)
        np.random.shuffle(original_dicts)

        threshold = int(len(original_dicts) * label_percentage / 100)
        original_dicts = original_dicts[:threshold]
    logger.info("{} original dicts used.".format(len(original_dicts)))
    
    dict_list = []
    for dict in original_dicts:
        city_name = dict["image_id"].split("_")[0]
        anns_name = dict["image_id"].replace("leftImg8bit.png", "gtFinePanopticParts.tif")
        part_file = os.path.join(CITYSCAPES_DATASET_PART_ANNS, split, city_name, anns_name)
        if PathManager.exists(part_file):
            if path_only:
                dict_list.append((dict, part_file))
            else:
                object_instances, part_instances = load_object_and_parts(dict, part_file)
                if for_segmentation:
                    # for segmentation, each instance is saved in a separate dict. 
                    if len(part_instances) > 0:
                        for object_annotation, part_annotations in zip(object_instances, part_instances):
                            if len(part_annotations) > 0:
                                new_dict = copy.deepcopy(dict)
                                new_dict["annotations"] = [object_annotation]
                                new_dict["part_annotations"] = [part_annotations]
                                dict_list.append(new_dict)
                else:
                    if len(part_instances) > 0:
                        dict["annotations"] = object_instances 
                        dict["part_annotations"] = part_instances
                        dict_list.append(dict)

        logger.info("{} annotation dicts registered in total.".format(len(dict_list)))
        if debug and len(dict_list) > 10:
            return dict_list 
    
    return dict_list


def register_cityscapes_part(name: str,
                             images_dirname: str,
                             annotations_dirname: str,
                             split: str,
                             path_only=False,
                             label_percentage: int=100,
                             for_segmentation: bool=False,
                             debug=False, 
    ):  
    DatasetCatalog.register(
        name,
        lambda: load_cityscapes_object_part_instances(
            images_dirname,
            annotations_dirname,
            split, 
            path_only=path_only,
            label_percentage=label_percentage, 
            for_segmentation=for_segmentation,
            debug=debug,
        ),
    )
    MetadataCatalog.get(name).set(
        thing_classes=OBJECT_CLASSES,
        part_classes=PART_CLASSES,
        classes=PART_CLASSES,
        split=split,
    )


