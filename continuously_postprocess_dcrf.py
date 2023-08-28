# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as dcrf_utils
import time

from pycocotools import mask as coco_mask
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, Instances


def dense_crf(image, label, n_labels, p=0.7, t=10, sd1=3, sd2=20, sc=13, compat1=3, compat2=10):
    annotated_label = label.to(torch.int32).numpy()
    colors, labels = np.unique(annotated_label, return_inverse=True)

    c = image.shape[2]
    h = image.shape[0]
    w = image.shape[1]

    d = dcrf.DenseCRF2D(w, h, n_labels)
    U = dcrf_utils.unary_from_labels(labels, n_labels, gt_prob=p, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    feats = dcrf_utils.create_pairwise_gaussian(sdims=(sd1, sd1), shape=(h, w))
    d.addPairwiseEnergy(feats, compat=compat1, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    feats = dcrf_utils.create_pairwise_bilateral(sdims=(sd2, sd2), schan=(sc, sc, sc),
                                                 img=image,
                                                 chdim=2)
    d.addPairwiseEnergy(feats, compat=compat2,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(t)
    Q = np.array(Q).reshape((n_labels, h, w)).argmax(axis=0)

    return Q


def proposals_to_coco_json(binary_mask):
    """
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(binary_mask)
    if num_instance == 0:
        return []

    rles = [coco_mask.encode(np.array(part_mask[:, :, None], order="F", dtype="uint8"))[0]
            for part_mask in binary_mask]
    for rle in rles:
        # "counts" is an array encoded by coco_mask as a byte-stream. Python3's
        # json writer which always produces strings cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what
        # the pycocotools/_mask.pyx does).
        rle["counts"] = rle["counts"].decode("utf-8")

    return [{"segmentation": rle} for rle in rles]


def get_argparse():
    parser = argparse.ArgumentParser(description='Postprocess pseudo-labels')
    parser.add_argument('--parallel_job_id', type=int, default=-1)
    parser.add_argument('--num_parallel_jobs', type=int, default=-1)
    parser.add_argument('--dataset_name', type=str, default="imagenet_1k_train")
    parser.add_argument('--mining_metric', type=str, default="iou_based")
    parser.add_argument('--dist_metric', type=str, default="dot")
    parser.add_argument('--res', type=str, default="res3_res4")
    parser.add_argument('--num_k', type=int, default=4)
    parser.add_argument('--feat_norm', action="store_true", default=False)
    parser.add_argument('--debug', action="store_true")

    return parser.parse_args()

path_root = "pseudo_labels/proposal_generation/"

# dcrf is done on larger resolution for performance reason.
augs = [T.ResizeScale(min_scale=1.0, max_scale=1.0, target_height=640, target_width=640),
        T.FixedSizeCrop(crop_size=(640, 640)),
        ]

if __name__ == "__main__":
    args = get_argparse()
    source_root = os.path.join(path_root, args.dataset_name, "detic_based", "generated_proposals", args.res, "{}_{}".format(args.dist_metric, args.num_k))
    target_root = os.path.join(path_root, args.dataset_name, "detic_based", "generated_proposals_processed", args.res, "{}_{}".format(args.dist_metric, args.num_k))

    code_list = os.listdir(source_root)
    if args.num_parallel_jobs > 0:
        num_total_classes = len(code_list)
        num_classes_per_job = num_total_classes // args.num_parallel_jobs
        num_remaining_classes = num_total_classes - args.num_parallel_jobs * num_classes_per_job
        num_current_job_classes = num_classes_per_job

        start_i = num_current_job_classes * (args.parallel_job_id-1)
        end_i = num_current_job_classes * args.parallel_job_id
        if args.parallel_job_id == args.num_parallel_jobs:
            end_i = num_total_classes
        code_list = code_list[start_i:end_i]

    for code in code_list:
        if not os.path.exists(os.path.join(target_root, code)):
            os.makedirs(os.path.join(target_root, code))

    num_total = 0
    for code in code_list:
        num_total += len(os.listdir(os.path.join(source_root, code)))
    t0 = time.time()
    while True:
        count = 0
        for code in code_list:
            fname_list = os.listdir(os.path.join(source_root, code))
            for fname in fname_list:
                if not os.path.exists(os.path.join(target_root, code, fname)):
                    data = torch.load(os.path.join(source_root, code, fname), "cpu")
                    mask = data["part_masks"]
                    # mask = data["part_mask"]
                    if mask is not None:
                        # image = utils.read_image(data["file_path"], format="RGB")
                        image = utils.read_image(data["file_name"], format="RGB")
                        # Resizing
                        aug_input = T.AugInput(image)
                        aug_input, transforms = T.apply_transform_gens(augs, aug_input)

                        image = aug_input.image
                        bmask = []
                        for segm in mask:
                            bmask.append(coco_mask.decode(segm["segmentation"]))
                        bmask = torch.tensor(np.array(bmask))
                        assert image.shape[:2] == bmask.shape[1:], "tensor shapes do not match. ({} != {})"\
                                            .format(image.shape[:2],  bmask.shape[1:])

                        num_c = bmask.shape[0]
                        cmask = (bmask * (torch.arange(num_c) + 1)[:, None, None]).sum(0)
                        cmask = torch.tensor(dense_crf(image, cmask, num_c + 1))
                        o_cls = cmask.unique()
                        o_cls = o_cls[o_cls != 0]
                        bmask = torch.zeros(len(o_cls), *cmask.shape).bool()
                        for i, c in enumerate(o_cls):
                            bmask[i] = cmask == c
                        data["part_masks"] = proposals_to_coco_json(bmask)

                    if args.debug:
                        assert False, "debug. "

                    torch.save(data, os.path.join(target_root, code, fname))

                    if count % 1000 == 1:
                        print("{} ({:.2f} %) images processed on process {} ({:.2f} / image)"\
                        .format(count, count/num_total*100, args.parallel_job_id, (time.time()-t0)/count), flush=True)
                count += 1
