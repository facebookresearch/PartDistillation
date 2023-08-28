# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import numpy as np

from typing import Dict, List
from detectron2.data import MetadataCatalog
from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator
from detectron2.utils.comm import all_gather, synchronize
from scipy.optimize import linear_sum_assignment


class mIOU_Matcher(SemSegEvaluator):
    def __init__(
        self,
        dataset_name: str,
        num_classes: int=8,
        distributed: bool = True,
    ):
        self._logger = logging.getLogger("part_distillation")
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")

        metadata = MetadataCatalog.get(dataset_name)
        self.pred_num_classes = num_classes
        self.gt_num_classes = len(metadata.part_classes) if hasattr(metadata, "part_classes") \
                            else len(metadata.thing_classes)
        self._class_names = metadata.thing_classes
        self.n = max(self.gt_num_classes, self.pred_num_classes)
        self._logger.info("mIOU-matcher initialized (n:{}/gt:{}/pd:{})."\
                    .format(self.n, self.gt_num_classes, self.pred_num_classes))

    def reset(self):
        self._conf_matrix = {}
        self._classes_used = set()

    def process(self, inputs: List[Dict], outputs: Dict):
        for output_per_image in outputs:
            pred_instances = output_per_image["predictions"].to(self._cpu_device)
            gt_instances = output_per_image["gt_instances"].to(self._cpu_device)

            pred_masks = pred_instances.pred_masks
            pred_classes = pred_instances.pred_classes
            gt_masks = gt_instances.gt_masks
            gt_classes = gt_instances.gt_classes
            gt_object_class = output_per_image["gt_object_label"].item()

            assert pred_masks.shape[1:] == gt_masks.shape[1:], '{} != {}'.format(pred_masks.shape, gt_masks.shape)
            if gt_object_class not in self._conf_matrix:
                self._conf_matrix[gt_object_class] = np.zeros((self.n + 1, self.n + 1), \
                                                    dtype=np.float64)
            pd = self._binary_mask_to_semseg(pred_masks, pred_classes)
            gt = self._binary_mask_to_semseg(gt_masks, gt_classes)

            conf_matrix_i = np.bincount(
                (self.n + 1) * pd.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix[gt_object_class].size,
            ).reshape(self._conf_matrix[gt_object_class].shape)

            self._conf_matrix[gt_object_class] += conf_matrix_i
            self._classes_used.add(gt_object_class)


    def _binary_mask_to_semseg(self, masks, classes):
        semseg = torch.full(masks.shape[1:], fill_value=self.n)
        for i, c in enumerate(classes):
            semseg[torch.where(masks[i]==True)] = c
        return semseg


    def evaluate(self):
        self._logger.info("Start matching ...")
        matching_mapper_dict = {}
        if self._distributed:
            synchronize()
            _classes_used = set()
            classes_used_total = all_gather(self._classes_used)
            for cset in classes_used_total:
                _classes_used = _classes_used.union(cset)
            self._classes_used = _classes_used

            synchronize()
            for k in self._classes_used:
                if k not in self._conf_matrix:
                    self._conf_matrix[k] = np.zeros((self.n + 1, self.n + 1), \
                                                    dtype=np.float64)

            synchronize()
            for k in self._classes_used:
                conf_matrix_list = all_gather(self._conf_matrix[k])

                _conf_matrix = np.zeros_like(self._conf_matrix[k])
                for conf_matrix in conf_matrix_list:
                    _conf_matrix += conf_matrix

                matching_mapper_dict[k] = self.majority_voting(_conf_matrix)
        return matching_mapper_dict


    def majority_voting(self, _conf_matrix):
        return torch.tensor(_conf_matrix[:self.pred_num_classes, :self.gt_num_classes].argmax(axis=1))
