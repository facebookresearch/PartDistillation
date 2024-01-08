# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
import torch 
from detectron2.data import DatasetCatalog, MetadataCatalog
from typing import List

IMAGENET_1K_DATASET_PATH = "datasets/imagenet_1k/"
IMAGENET_22K_DATASET_PATH = "datasets/imagenet_22k/"
EXCLUDE_CODE_PATH = "datasets/metadata/exclude_code_list.pkl"
METADATA_PATH = "datasets/metadata/"

def load_multiple_imagenet_images(filename_list, dataset_path_list, min_object_area_ratio, class_code_to_class_id, path_only=False, debug=False):
    logger = logging.getLogger("part_distillation")
    logger.info("Start loading multiple imagenet data.")

    dict_list = []
    for dataset_path in dataset_path_list:
        dict_list.extend(load_imagenet_images(filename_list, dataset_path, min_object_area_ratio, class_code_to_class_id, path_only, debug))

    logger.info("Total dataset loaded: {}\n".format(len(dict_list)))

    return dict_list



def load_imagenet_images(filename_list, dataset_path, min_object_area_ratio, class_code_to_class_id, path_only=False, debug=False):
    logger = logging.getLogger("part_distillation")
    logger.info("Start loading imagenet data images and proposals from {}.".format(dataset_path))

    dict_list = []
    count = 0 
    used  = 0
    filename_list = [fname for fname in filename_list if fname in os.listdir(dataset_path)]
    if debug:
        filename_list = filename_list[:100]
    for fname in filename_list:
        ann_list = os.listdir(os.path.join(dataset_path, fname))
        if debug:
            ann_list = ann_list[:10]
        for ann_name in ann_list:
            count += 1
            ann_path = os.path.join(dataset_path, fname, ann_name)
            if os.path.exists(ann_path):
                if path_only:
                    dict_list.append(tuple([dataset_path, fname, ann_name]))
                    used += 1
                else:
                    try:
                        ann_dict = torch.load(ann_path)
                    except EOFError:
                        print(ann_path, " is corrupted.", flush=True)
                        continue 
                    if ann_dict["object_ratio"] > min_object_area_ratio:
                        new_dict = {"file_name": ann_dict["file_path"],
                                    "image_id": ann_dict["file_name"],
                                    "class_code": fname,
                                    "gt_object_class": class_code_to_class_id[fname],
                                    "height": None,
                                    "width": None,
                                    "pseudo_annotations": []}
                        if ann_dict["part_mask"] is None:
                            continue 
                        for segm in ann_dict["part_mask"]:
                            new_dict["pseudo_annotations"].append({"segmentation": segm["segmentation"]})
                            height, width = segm["segmentation"]["size"]
                            new_dict["height"] = height 
                            new_dict["width"]  = width
                        if len(new_dict["pseudo_annotations"]) > 0:
                            dict_list.append(new_dict)
                            used += 1
    logger.info("Dataset loaded ({}/{})".format(used, count))

    return dict_list


def register_imagenet_with_proposals(
    name: str,
    dataset_path: str,
    split: str,
    min_object_area_ratio: float=-1.0, 
    partitioned_imagenet: bool=False,
    total_partitions: int=10, 
    partition_index: int=0,
    dataset_path_list=[],
    filtered_code_path_list: List[str]=[""],
    exclude_code_path: str="",
    single_class_code: str="",
    path_only: bool=False,
    debug=False, 
):  
    logger = logging.getLogger("part_distillation")
    logger.info("Start registering imagenet with proposals.")
    if "1k" in name:
        imagenet_size = "1k"
        dataset_path = IMAGENET_1K_DATASET_PATH + "train"
        fname_to_classname = torch.load(os.path.join(METADATA_PATH, 'imagenet_1k_fname_classname_dict.pkl'))
    elif "22k" in name:
        with open(os.path.join(IMAGENET_22K_DATASET_PATH, "synsets.dat"), "r") as f:
            class_code_list = f.readlines()
        class_code_list = [_.strip() for _ in class_code_list]
        with open(os.path.join(IMAGENET_22K_DATASET_PATH, "words.txt"), "r") as f:
            fname_cname_pair_list = f.readlines()
        fname_to_classname = {x.split('\t')[0]: x.split('\t')[1].strip() for x in fname_cname_pair_list}
        fname_to_classname = {k:v for k, v in fname_to_classname.items() if k in class_code_list}
    
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

    if len(dataset_path_list) == 0:
        dataset_path_list = [dataset_path]
    
    DatasetCatalog.register(
    name,
    lambda: load_multiple_imagenet_images(
        filename_list=list(fname_to_classname.keys()),
        dataset_path_list=dataset_path_list,
        min_object_area_ratio=min_object_area_ratio,
        class_code_to_class_id=class_code_to_class_id,
        path_only=path_only,
        debug=debug,
        ),
    )

    MetadataCatalog.get(name).set(
        classes=list(fname_to_classname.values()),
        class_codes=list(fname_to_classname.keys()),
        fname_to_classname=fname_to_classname,
        class_code_to_class_id=class_code_to_class_id,
        split=split,
    )

