# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging
import numpy as np
import torch
import detectron2.utils.comm as comm

from detectron2.utils.comm import is_main_process, synchronize, all_gather
from detectron2.evaluation import DatasetEvaluator
from sklearn.cluster import KMeans


class ClusteringModule(DatasetEvaluator):
    def __init__(self,
                 distributed=True,
                 num_clusters=8,
                 ):
        self._logger = logging.getLogger("part_distillation")
        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self.num_clusters = num_clusters
        self.kmeans_module = KMeans(n_clusters=self.num_clusters, random_state=0)


    def reset(self):
        self._proposal_features = []
        self._class_labels_list = []


    def process(self, inputs, outputs):
        for output_per_image in outputs:
            proposals = output_per_image["proposal_features"].to(self._cpu_device)
            gt_label = output_per_image["gt_label"].to(self._cpu_device)
            self._proposal_features.append(proposals)
            self._class_labels_list.append(gt_label)


    def evaluate(self):
        if self._distributed:
            synchronize()
            proposal_features = all_gather(self._proposal_features)
            proposal_features = list(itertools.chain(*proposal_features))

            gt_labels = all_gather(self._class_labels_list)
            gt_labels = list(itertools.chain(*gt_labels))

        proposal_features = torch.cat(proposal_features, dim=0)
        gt_labels = torch.cat(gt_labels, dim=0)
        gt_unique = gt_labels.unique().long().numpy()

        # only run the main process since clustering is on cpu.
        cluster_centroids_dict = {}
        if comm.is_main_process():
            for cid in gt_unique:
                proposal_features_i = proposal_features[gt_labels == cid]
                if proposal_features_i.shape[0] > self.num_clusters:
                    cluster_centroids_dict[cid] = self._get_cluster_centroids(proposal_features_i, cid)
                else:
                    cluster_centroids_dict[cid] = torch.randn(self.num_clusters, proposal_features_i.shape[1])

        synchronize()
        cluster_centroids_dict = all_gather(cluster_centroids_dict)
        cluster_centroids_dict = cluster_centroids_dict[0] # 0 is the main process.

        return copy.deepcopy(cluster_centroids_dict)


    def _get_cluster_centroids(self, proposal_features, cid):
        kmeans = self.kmeans_module.fit(proposal_features)
        cpreds = kmeans.labels_
        cpreds = torch.tensor(cpreds)

        centroids = kmeans.cluster_centers_
        centroids = torch.tensor(centroids).float()

        return centroids
