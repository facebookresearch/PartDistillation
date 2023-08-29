# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import torch 
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

PART_IMAGENET_ANNOTATION_ROOT = "datasets/part_imagenet/"
IMAGENET_IMAGE_DIRNAME = "datasets/imagenet_1k/train/"

def load_json_with_label_limit(json_file, image_root, name, label_percentage):
    logger = logging.getLogger("part_distillation")
    logger.info("Starting loading part imagenet data")

    dict_list = load_coco_json(json_file, image_root, name)
    if label_percentage < 100:
        # shuffle and pick the first n.
        np.random.seed(1234)
        np.random.shuffle(dict_list)

        threshold = int(len(dict_list) * label_percentage / 100)
        dict_list = dict_list[:threshold]
    logger.info("{} annotation dicts registered in total.".format(len(dict_list)))

    return dict_list


def register_part_imagenet(name,
                           images_dirname,
                           annotations_dirname,
                           split,
                           label_percentage: int=100,
                           debug=False, 
    ):  
    assert isinstance(name, str), name
    assert isinstance(images_dirname, str), images_dirname
    assert isinstance(annotations_dirname, str), annotations_dirname

    if len(images_dirname) == 0:
        images_dirname = IMAGENET_IMAGE_DIRNAME 
    if len(annotations_dirname) == 0:
        annotations_dirname = PART_IMAGENET_ANNOTATION_ROOT
    json_file = os.path.join(annotations_dirname, split + ".json")
    DatasetCatalog.register(name, lambda: load_json_with_label_limit(json_file, images_dirname, name, label_percentage))
    
    # class id is defined based on imagenet-1k idexing.
    fname_to_classname = torch.load('datasets/metadata/imagenet_1k_fname_classname_dict.pkl')
    class_code_to_class_id = {k: i for i, k in enumerate(list(fname_to_classname.keys()))}    
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=images_dirname,
                                  imagenet_1k_class_code_to_class_id=class_code_to_class_id
                                  )


