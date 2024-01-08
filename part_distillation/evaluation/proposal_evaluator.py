# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import itertools
import copy 
import json
import logging
import numpy as np
import torch
import time 

from collections import OrderedDict
from typing import Dict, List
from detectron2.utils.comm import is_main_process, synchronize, gather
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils.logger import create_small_table
from pycocotools import mask as maskUtils

__all__ = ["ProposalEvaluator"]

def pairwise_mask_iou_cocoapi(pr_masks, gt_masks):
    gt_masks = [maskUtils.encode(np.asfortranarray(m.numpy())) for m in gt_masks]
    ious = maskUtils.iou(pr_masks, gt_masks, [0 for _ in range(len(gt_masks))])

    return torch.tensor(ious)


def _evaluate_box_proposals(proposals_list, gt_masks_list, area="all", limit=None, thresholds=None):
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0**2, 1e5**2],  # all
        [0**2, 32**2],  # small
        [32**2, 96**2],  # medium
        [96**2, 1e5**2],  # large
        [96**2, 128**2],  # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2],
    ]  # 512-inf

    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = [] 
    num_pos = 0 
    time_for_iou = []

    for (proposals_mask, proposals_score), gt_masks in zip(proposals_list, gt_masks_list): 
        inds = proposals_score.sort(descending=True)[1]

        proposals = [proposals_mask[i] for i in inds]

        if len(proposals) == 0 or len(gt_masks) == 0:
            continue 
        gt_areas = gt_masks.float().flatten(1).sum(-1)
        valid_gt_inds = (gt_areas > area_range[0]) & (gt_areas <= area_range[1]) 
        gt_masks = gt_masks[valid_gt_inds]

        num_pos += len(gt_masks)

        if len(gt_masks) == 0:
            continue 
        
        if limit is not None and len(proposals) > limit:
            proposals = proposals[:limit] 
        
        t1 = time.time() 
        overlaps = pairwise_mask_iou_cocoapi(proposals, gt_masks)  # 20x faster.
        time_for_iou.append(time.time()-t1)

        _gt_overlaps = torch.zeros(len(gt_masks))
        for j in range(min(len(proposals), len(gt_masks))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert np.isclose(_gt_overlaps[j], gt_ovr, atol=1e-5), "{} | {}".format(_gt_overlaps[j], gt_ovr)
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1
        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
        "avg_time_for_iou": np.mean(time_for_iou)
    }



class ProposalEvaluator(DatasetEvaluator):
    def __init__(
        self,
        distributed: bool = True,
        output_dir: str = None,
        areas: List[str] = ["small", "medium", "large", "all"],
        limit: int=-1,
    ):
        """
        This evaluator evaluates baseline methods. 

        The evaluation is on AR metric. 

        This evaluator will handle subset_class evaluation as well. 
        """
        self._logger = logging.getLogger(__name__)

        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self.areas = areas 
        self.limit = limit

    def reset(self):
        self._predictions = []
        self._gts = []

    def process(self, inputs: List[Dict], outputs: Dict):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is a dictionary which contains
                     "gt_part_masks" : binary masks of gt ordered according to "gt_part_classes"
                     "pred_part_masks" : binary masks of prediction ordered according to "pred_part_classes"
        """
        for output_per_image in outputs:
            proposals = output_per_image["proposals"].to(self._cpu_device)
            part_masks_gt = output_per_image["gt_masks"].to(self._cpu_device)
            proposals_mask = [maskUtils.encode(np.asfortranarray(m[0].numpy())) for m in proposals.pred_masks.split(1)]
            proposals_score = proposals.scores
            self._predictions.append((proposals_mask, proposals_score))
            self._gts.append(part_masks_gt.gt_masks)


    def evaluate(self):
        print("Evaluate starts. ", flush=True)
        if self._distributed:
            synchronize()
            predictions = gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            gts = gather(self._gts, dst=0)
            gts = list(itertools.chain(*gts))

            if not is_main_process():
                return {} 
        else:
            predictions = self._predictions 
            gts = self._gts 

        if len(predictions) == 0:
            self._logger.warning("[ProposalEvaluator] Did not receive valid predictions.")
            return {}
        
        self._results = OrderedDict()
        self._eval_proposals(predictions, gts)

        return copy.deepcopy(self._results)


    def _eval_proposals(self, predictions, gts):
        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        # NOTE: Don't evaluate sizes until we redefine the thresholds.
        areas = {"all": ""}#, "small": "s", "medium": "m", "large": "l"}
        for limit in [1, 10, 50, 100, 200]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, gts, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
                res["# instances"] = len(gts)
                recalls_i = stats["recalls"]
                recalls_i = "   ".join(["{:.2f}".format(r*100) for r in recalls_i])
                print("recall@{}: ".format(limit), recalls_i, flush=True)
                print("AR@{}: ".format(limit), "{:.2f}".format(res[key]), flush=True)
        self._logger.info("Proposal metrics: \n\n{}\n\n".format(create_small_table(res)))
        self._results["box_proposals"] = res
