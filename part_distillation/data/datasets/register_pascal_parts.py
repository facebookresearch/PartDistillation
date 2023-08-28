# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import logging
import os
import copy
import numpy as np
import pycocotools.mask as mask_util
import scipy.io
from typing import Any, Dict, List, Tuple, Union
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.pascal_voc import CLASS_NAMES, load_voc_instances
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from .pascal_info import get_orig_part, categories


OBJ_NAMES_TO_PART_NAMES_DICT = categories
PASCALPARTS_DATASET_PATH = "datasets/pascal_parts/images/"
PASCALPARTS_ANNOTATION_PATH = "datasets/pascal_parts/annotations/"


def mask_to_bbox(mask: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Returns the x1, y1, x2, y2 of the tightest bounding box around
    the given binary mask.
    """
    indices = np.where(mask)
    y1, y2 = np.amin(indices[0]), np.amax(indices[0])
    x1, x2 = np.amin(indices[1]), np.amax(indices[1])

    return x1, y1, x2, y2


def get_part_annotation_dict(part_instance: Any, subset_class_names: Union[List[str], Any],
                             encode=True, subset_part_name_to_ids={}) -> Tuple[Dict, List]:
    class_name = part_instance[0][0]
    if class_name == "table":
        class_name = "diningtable"
    object_dict = {
        "object_category": class_name,
        "category_id": subset_class_names.index(class_name),
        "segmentation": mask_util.encode(part_instance[2]),
        "bbox": mask_to_bbox(part_instance[2]),
        "bbox_mode": BoxMode.XYXY_ABS,
    }
    parts = part_instance[3][0] if part_instance[3].shape[0] > 0 else []

    part_instances = []
    for p in parts:
        part_name = p[0][0].split("_")[0]
        orig_part_name = get_orig_part(class_name, part_name)
        part_instances.append(
            {
                "part_category": p[0][0],
                "orig_part_category": orig_part_name,
                "orig_part_category_id": subset_part_name_to_ids[orig_part_name],
                "bbox": mask_to_bbox(p[1]),
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": mask_util.encode(p[1]) if encode else p[1],
            }
        )
    return object_dict, part_instances



def load_pascal_parts_instances(
    images_dirname: str,
    annotations_dirname: str,
    split: str,
    subset_class_names: Union[List[str], Any],
    subset_part_name_to_ids: dict,
    label_percentage: int,
    for_segmentation: bool,
    debug: bool,
) -> List[Dict]:
    voc_original_dicts = load_voc_instances(images_dirname,
                                            split,
                                            CLASS_NAMES,
                                            )
    if debug:
        voc_original_dicts = voc_original_dicts[:100]

    part_annotations_localdir = PathManager.get_local_path(annotations_dirname)
    logger = logging.getLogger("part_distillation")
    logger.info("Starting loading pascal parts data")
    num_done, num_found = 0, 0
    final_dicts = []
    for dict in voc_original_dicts:
        dict["part_annotations"] = []
        fileid = dict["image_id"]
        num_done += 1
        part_file = os.path.join(part_annotations_localdir, fileid + ".mat")
        dict["part_annotation_file"] = part_file
        if PathManager.exists(part_file):
            instances = scipy.io.loadmat(part_file)["anno"][0, 0][1][0]
            dict["annotations"] = []
            for inst in instances:
                if inst[0][0] in subset_class_names:
                    object_annotation, part_annotations = get_part_annotation_dict(inst,
                                                            subset_class_names=subset_class_names,
                                                            subset_part_name_to_ids=subset_part_name_to_ids)

                    # for segmentation, each instance is saved in a separate dict.
                    if for_segmentation:
                        new_dict = copy.deepcopy(dict)

                        # some object has no parts.
                        if len(part_annotations) > 0:
                            new_dict["annotations"].append(object_annotation)
                            new_dict["part_annotations"].append(part_annotations)
                            final_dicts.append(new_dict)
                    else:
                        if len(part_annotations) > 0:
                            dict["annotations"].append(object_annotation)
                            dict["part_annotations"].append(part_annotations)
            logging.info("Num done = %d/%d, num with parts = %d" % (num_done, len(voc_original_dicts), num_found))

            if len(dict["part_annotations"]) > 0 and not for_segmentation:
                final_dicts.append(dict)
                num_found += 1

    if label_percentage < 100:
        # shuffle and pick first n.
        np.random.seed(1234)
        np.random.shuffle(final_dicts)

        threshold = int(len(final_dicts) * label_percentage / 100)
        final_dicts = final_dicts[:threshold]
    logger.info("{} annotation dicts registered in total.".format(len(final_dicts)))

    return final_dicts


def register_pascal_parts(
    name: str,
    images_dirname: str,
    annotations_dirname: str,
    split: str,
    year: int=2012,
    subset_class_names=None,
    label_percentage: int=100,
    for_segmentation: bool=False,
    debug=False,
):
    """
    subset_class_names: Subset of PascalParts classes to use,
    label_percentage: Percentage of labels to register. Used for few-shot learning.
    for_segmentation: For segmentation evaluation, each image has one object instance.
                      Dataset will then have duplicate images.
    debug: For quick dubugging, only register a small portion.
    """
    if len(images_dirname) == 0:
        images_dirname = PASCALPARTS_DATASET_PATH
    if len(annotations_dirname) == 0:
        annotations_dirname = PASCALPARTS_ANNOTATION_PATH
    if subset_class_names is not None and len(subset_class_names) > 0:
        subset_class_names = sorted(subset_class_names)
    else:
        subset_class_names = CLASS_NAMES

    pid = 0
    subset_part_name_to_ids = {}
    for class_name in subset_class_names:
        if class_name == "table":
            class_name = "diningtable"

        # part IDs are re-defined for subset classes.
        for part in OBJ_NAMES_TO_PART_NAMES_DICT[class_name]:
            pname = part.orig_name
            if pname not in subset_part_name_to_ids:
                subset_part_name_to_ids[pname] = pid
                pid += 1

    DatasetCatalog.register(
        name,
        lambda: load_pascal_parts_instances(
            images_dirname, annotations_dirname, split,
            subset_class_names=subset_class_names,
            subset_part_name_to_ids=subset_part_name_to_ids,
            label_percentage=label_percentage,
            for_segmentation=for_segmentation,
            debug=debug,
        ),
    )
    MetadataCatalog.get(name).set(
        thing_classes=list(subset_class_names),
        part_classes=list(subset_part_name_to_ids.keys()),
        classes=list(subset_part_name_to_ids.keys()),
        dirname=images_dirname,
        annotations_dirname=annotations_dirname,
        year=year,
        split=split,
    )
