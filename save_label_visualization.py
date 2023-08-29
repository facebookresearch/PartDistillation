# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
This file is to sanity-check the saved visualization. 
"""

import os 
import torch 
import numpy as np 
from pycocotools import mask as coco_mask
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances

rootpath = "pseudo_labels/part_labels/processed_proposals/human-only-0.3/imagenet_1k_train/detic/res3_res4/dot_4_norm_False/" 
augs = [T.ResizeScale(min_scale=1.0, max_scale=1.0, target_height=640, target_width=640)]
targetpath = "visualization/"
if __name__ == "__main__":
    cname_list = os.listdir(rootpath)
    path_list  = [os.path.join(rootpath, c, f) for c in cname_list for f in os.listdir(os.path.join(rootpath, c))]
    np.random.shuffle(path_list)
    path_list  = path_list[:100]

    for path in path_list:
        data  = torch.load(path, "cpu")
        image = utils.read_image(data["file_path"], format="RGB")
        image = T.apply_transform_gens(augs, T.AugInput(image))[0].image
        data['image'] = image 

        torch.save(data, os.path.join(targetpath, data['file_name']))

