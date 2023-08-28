# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import torch

from typing import List
from detectron2.data import DatasetCatalog, MetadataCatalog

IMAGENET_1K_DATASET_PATH = "datasets/imagenet_1k/"
IMAGENET_22K_DATASET_PATH = "datasets/imagenet_22k/"
PART_IMAGENET_CLASSES_TRAIN = os.listdir("datasets/part_imagenet/train")
PART_IMAGENET_CLASSES_VAL = os.listdir("datasets/part_imagenet/val")
PART_IMAGENET_CLASSES_TEST = os.listdir("datasets/part_imagenet/test")


def load_imagenet_images(fname_to_cname_dict,
                         dataset_path, split,
                         class_code_to_class_id,
                         save_path,
                         with_given_mask=False,
                         object_mask_path="",
                         debug=False):
    logger = logging.getLogger("part_distillation")
    logger.info("Starting loading imagenet data.")

    dict_list = []
    done_already = 0
    total_num = 0
    filename_list = [fname for fname in fname_to_cname_dict.keys() if fname in os.listdir(dataset_path)]
    if debug:
        filename_list = filename_list[:100]
    for fname in filename_list:
        image_list = os.listdir(os.path.join(dataset_path, fname))
        if debug:
            image_list = image_list[:10]
        for iname in image_list:
            total_num += 1
            if not os.path.exists(os.path.join(save_path, fname, iname)):
                data = {"file_path": os.path.join(dataset_path, fname, iname),
                        "file_name": iname,
                        "class_code": fname,
                        "gt_object_class": class_code_to_class_id[fname],
                        "class_name": fname_to_cname_dict[fname]}

                if with_given_mask:
                    if os.path.exists(os.path.join(object_mask_path, fname, iname)):
                        object_data = torch.load(os.path.join(object_mask_path, fname, iname))
                        if len(object_data["object_masks"]) > 0:
                            # object masks are ordered by confidence already (use most confident mask).
                            data["pseudo_annotations"] = [{"segmentation" : object_data["object_masks"][0]["segmentation"]}]
                            dict_list.append(data)
                else:
                    dict_list.append(data)
            else:
                done_already += 1
    logger.info("Progress: {}/{} ({} to go!)".format(done_already, total_num, len(dict_list)))

    return dict_list



def register_imagenet(
    name: str,
    split: str,
    partitioned_imagenet: bool=True,
    total_partitions: int=10,
    partition_index: int=0,
    save_path: str="",
    with_given_mask:bool=False,
    object_mask_path: str="",
    filtered_code_path_list: List[str]=[""],
    exclude_code_path: str="",
    single_class_code: str="",
    use_part_imagenet_classes: bool=False,
    debug=False,
):
    logger = logging.getLogger("part_distillation")
    logger.info("Start registering imagenet dataset.")
    if "1k" in name:
        imagenet_size = "1k"
        dataset_path = IMAGENET_1K_DATASET_PATH + "train"
        with open(os.path.join(IMAGENET_1K_DATASET_PATH, "labels.txt"), "r") as f:
            fname_cname_pair_list = f.readlines()
        fname_to_classname = {x.split(',')[0]: x.split(',')[1].strip() for x in fname_cname_pair_list}
    elif "22k" in name:
        imagenet_size = "22k"
        dataset_path = IMAGENET_22K_DATASET_PATH
        with open(os.path.join(IMAGENET_22K_DATASET_PATH, "synsets.dat"), "r") as f:
            class_code_list = f.readlines()
        class_code_list = [_.strip() for _ in class_code_list]
        with open(os.path.join(IMAGENET_22K_DATASET_PATH, "words.txt"), "r") as f:
            fname_cname_pair_list = f.readlines()
        fname_to_classname = {x.split('\t')[0]: x.split('\t')[1].strip() for x in fname_cname_pair_list}
        fname_to_classname = {k:v for k, v in fname_to_classname.items() if k in class_code_list}
    elif use_part_imagenet_classes:
        PART_IMAGENET_CLASSES = []
        if "val" in split:
            PART_IMAGENET_CLASSES += PART_IMAGENET_CLASSES_VAL
        if "train" in split:
            PART_IMAGENET_CLASSES += PART_IMAGENET_CLASSES_TRAIN
        if "test" in split:
            PART_IMAGENET_CLASSES += PART_IMAGENET_CLASSES_TEST
        fname_to_classname = {k:v for k, v in fname_to_classname.items() if k in PART_IMAGENET_CLASSES}
        logger.info("Registering {} PartImageNet Classes.".format(len(PART_IMAGENET_CLASSES)))
    else:
        raise ValueError("{} not supported.".format(name))

    # Use subset classes.
    for filtered_code_path in filtered_code_path_list:
        if len(filtered_code_path) > 0:
            filtered_code_list = torch.load(filtered_code_path)
            fname_to_classname = {k:v for k, v in fname_to_classname.items() if k in filtered_code_list}
    if len(single_class_code) > 0:
        fname_to_classname = {k: v for k, v in fname_to_classname.items() if k == single_class_code}
    if len(exclude_code_path) > 0:
        exclude_code_list = torch.load(exclude_code_path)
        fname_to_classname = {k:v for k, v in fname_to_classname.items() if k not in exclude_code_list}
    class_code_to_class_id = {k: i for i, k in enumerate(list(fname_to_classname.keys()))}

    key_list_all = list(fname_to_classname.keys())
    if partitioned_imagenet:
        # Parallelize the preprocessing.
        partition_size = len(key_list_all) // total_partitions
        start_i = partition_index * partition_size
        end_i = (partition_index+1) * partition_size if partition_index + 1 < total_partitions else len(list(fname_to_classname.keys()))
        key_list = list(key_list_all)[start_i: end_i]
        fname_to_classname = {k: fname_to_classname[k] for k in key_list}
    logger.info("{}/{} classes used.".format(len(fname_to_classname), len(key_list_all)))

    DatasetCatalog.register(
        name,
        lambda: load_imagenet_images(
            fname_to_cname_dict=fname_to_classname,
            dataset_path=dataset_path,
            split=split,
            class_code_to_class_id=class_code_to_class_id,
            save_path=save_path,
            with_given_mask=with_given_mask,
            object_mask_path=object_mask_path,
            debug=debug,
        ),
    )

    MetadataCatalog.get(name).set(
        classes=list(fname_to_classname.values()),
        class_codes=list(fname_to_classname.keys()),
        fname_to_classname=fname_to_classname,
        class_code_to_class_id=class_code_to_class_id,
        save_path=save_path,
        split=split,
    )
