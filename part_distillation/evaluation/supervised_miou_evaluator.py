# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import numpy as np
import torch

from typing import Dict, List
from detectron2.data import MetadataCatalog
from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator
from detectron2.utils.comm import all_gather, synchronize
from detectron2.utils.logger import create_small_table

class Supervised_mIOU_Evaluator(SemSegEvaluator):
    def __init__(
        self,
        dataset_name: str,
        num_classes: int=8,
        distributed: bool = True,
        output_dir: str = None,
    ):
        self._logger = logging.getLogger(__name__)

        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        self._cpu_device = torch.device("cpu")
        self._num_classes = num_classes
        self._class_names =  MetadataCatalog.get(dataset_name).thing_classes
        self._logger.info("Supervised mIOU-evaluator initialized (gt:{}).".format(self._num_classes))

    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes+1, self._num_classes+1))

    def process(self, inputs: List[Dict], outputs: Dict):
        for output_per_image in outputs:
            pred_instances = output_per_image["predictions"].to(self._cpu_device)
            gt_instances = output_per_image["gt_instances"].to(self._cpu_device)

            pred_masks = pred_instances.pred_masks
            pred_classes = pred_instances.pred_classes
            gt_masks = gt_instances.gt_masks
            gt_classes = gt_instances.gt_classes

            assert pred_masks.shape[1:] == gt_masks.shape[1:]
            pd = self._binary_mask_to_semseg(pred_masks, pred_classes)
            gt = self._binary_mask_to_semseg(gt_masks, gt_classes)

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pd.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)


    def _binary_mask_to_semseg(self, masks, classes):
        semseg = torch.full(masks.shape[1:], fill_value=self._num_classes)
        for i, c in enumerate(classes):
            semseg[torch.where(masks[i]==True)] = c
        return semseg

    def evaluate(self):
        seg_results_all = {"mIoU": [],
                           "mACC": [],
                           "mIoPred": [],
                           }
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)

            _conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                _conf_matrix += conf_matrix

            seg_results = self.measure_mIOU(_conf_matrix)
            seg_results_all["mIoU"].extend([v for k, v in seg_results.items() if "IoU-" in k and not np.isnan(v)])
            seg_results_all["mACC"].append(seg_results["mACC"])
            seg_results_all["mIoPred"].append(seg_results["mIoPred"])

        seg_results_all["mIoU"] = np.mean(seg_results_all["mIoU"])
        seg_results_all["mIoPred"] = np.mean(seg_results_all["mIoPred"])
        seg_results_all["mACC"] = np.mean(seg_results_all["mACC"])
        return seg_results_all


    def measure_mIOU(self, conf_matrix):
        """
        Args:
        conf_matrix: nd.array for confusion matrix corressponding
            to sematic segmentation
        num_classes: number of foreground classes in the dataset;
            bg considered separately
        class_names: List with names of forground classes
        """
        num_classes = self._num_classes
        class_names = self._class_names

        acc = np.full(num_classes, np.nan, dtype=float)
        iou = np.full(num_classes, np.nan, dtype=float)
        # precision or intersection over prediction
        iopred = np.full(num_classes, np.nan, dtype=float)
        tp = conf_matrix.diagonal()[:-1].astype(float)
        pos_gt = np.sum(conf_matrix[:, :-1], axis=0).astype(float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(conf_matrix[:-1, :], axis=1).astype(float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        iopred_valid = pos_pred > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        iopred[iopred_valid] = tp[iopred_valid] / pos_pred[iopred_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        miopred = np.sum(iopred[iopred_valid]) / np.sum(iopred_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)
        res = {}
        res["mIoU"] = 100 * miou
        res["mIoPred"] = 100 * miopred
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(class_names):
            res[f"IoU-{name}"] = 100 * iou[i]
        for i, name in enumerate(class_names):
            res[f"IoPred-{name}"] = 100 * iopred[i]
        for i, name in enumerate(class_names):
            res[f"ACC-{name}"] = 100 * acc[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(class_names):
            res[f"ACC-{name}"] = 100 * acc[i]

        self._logger.info("Supervised mIOU evaluation: \n\n{}\n\n".format(create_small_table(res)))
        return res
