# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import warnings
warnings.filterwarnings('ignore', category=UserWarning)
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import logging
import os
import sys 
import wandb 
import torch
import torch.nn as nn 
import numpy as np 
import detectron2.utils.comm as comm

sys.path.append('Detic/third_party/CenterNet2')
sys.path.append('Detic/third_party/Deformable-DETR')

from collections import OrderedDict
from pathlib import Path

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog, 
                             build_detection_test_loader, 
                             build_detection_train_loader)

from detectron2.engine import (DefaultTrainer,
                               default_argument_parser,
                               default_setup,
                               launch)

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import is_main_process, synchronize
from detectron2.evaluation import verify_results, inference_on_dataset, print_csv_format

from part_distillation import (add_maskformer2_config, 
                               add_wandb_config, 
                               add_part_ranking_config, 
                               add_custom_datasets_config)

from part_distillation.data.dataset_mappers.proposal_dataset_mapper import ProposalDatasetMapper
from part_distillation.data.dataset_mappers.voc_parts_mapper import VOCPartsMapper
from part_distillation.data.dataset_mappers.cityscapes_part_mapper import CityscapesPartMapper
from part_distillation.data.dataset_mappers.part_imagenet_mapper import PartImageNetMapper
from part_distillation.data.dataset_mappers.imagenet_part_ranking_dataset_mapper import ImagenetPartRankingDatasetMapper

from part_distillation.data.datasets.register_pascal_parts import register_pascal_parts
from part_distillation.data.datasets.register_cityscapes_part import register_cityscapes_part
from part_distillation.data.datasets.register_part_imagenet import register_part_imagenet
from part_distillation.data.datasets.register_imagenet_with_proposals import register_imagenet_with_proposals
from part_distillation.data.datasets.register_imagenet import register_imagenet

from part_distillation.evaluation.clustering_module import ClusteringModule
from part_distillation.evaluation.null_evaluator import NullEvaluator
from part_distillation.evaluation.miou_evaluator import mIOU_Evaluator
from part_distillation.evaluation.miou_matcher import mIOU_Matcher

from base_trainer import maybe_dp, get_mode



class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(self, cfg, dataset_name):
        if "pre_labeling" in dataset_name: 
            return ClusteringModule(num_clusters=cfg.PART_RANKING.NUM_CLUSTERS)
        elif "post_labeling" in dataset_name:
            return NullEvaluator()
        elif "match" in dataset_name:
            return mIOU_Matcher(dataset_name, 
                                num_classes=cfg.PART_RANKING.NUM_CLUSTERS)
        elif "evaluate" in dataset_name:
            return mIOU_Evaluator(dataset_name)

    @classmethod
    def build_test_loader(self, cfg, dataset_name):
        if "pascal" in dataset_name:
            mapper = VOCPartsMapper(cfg, is_train=False)
        elif "part_imagenet" in dataset_name:
            mapper = PartImageNetMapper(cfg, dataset_name, is_train=False)
        elif "cityscapes" in dataset_name:
            mapper = CityscapesPartMapper(cfg, is_train=False)
        elif "imagenet" in dataset_name:    
            class_code_to_class_index = MetadataCatalog.get(dataset_name).class_code_to_class_id
            mapper = ImagenetPartRankingDatasetMapper(cfg, class_code_to_class_index)

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


    @classmethod
    def test(cls, cfg, model):
        logger  = logging.getLogger("part_distillation")
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            
            # set mode
            mode = get_mode(dataset_name)
            maybe_dp(model).mode = mode
            maybe_dp(model).register_metadata(dataset_name)
            logger.info("Starting {} mode on {}.".format(mode, dataset_name))

            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = cls.build_evaluator(cfg, dataset_name)
            results_i = inference_on_dataset(model, data_loader, evaluator)
            
            if mode == "cluster":
                maybe_dp(model).register_classifier(results_i) 
                logger.info("Cluster centroids are registered as classifiers ({} classes).".format(len(results_i)))
                continue 
            elif mode == "match":
                maybe_dp(model).update_majority_vote_mapping(results_i)
                logger.info("Majority vote result:\n{}".format(results_i))
                continue 
  
            results.update(results_i)
            if comm.is_main_process():
                assert isinstance(results_i, dict), \
                "Evaluator must return a dict on the main process. Got {} instead.".format(results_i)
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
            comm.synchronize()

        if len(results) == 1:
            results = list(results.values())[0]
        
        comm.synchronize()
        if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
            wandb.log(results)

        return results
        

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_part_ranking_config(cfg)
    add_custom_datasets_config(cfg)
    add_wandb_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="part_distillation")

    # register dataset
    for dataset_name in cfg.DATASETS.TEST:
        if "part_imagenet" in dataset_name:
            register_part_imagenet(name=dataset_name, 
                                   images_dirname=cfg.CUSTOM_DATASETS.PART_IMAGENET.IMAGES_DIRNAME,
                                   annotations_dirname=cfg.CUSTOM_DATASETS.PART_IMAGENET.ANNOTATIONS_DIRNAME,
                                   split=dataset_name.split('_')[-1],
                                   debug=cfg.CUSTOM_DATASETS.PART_IMAGENET.DEBUG,
                                    )

        elif "cityscapes" in dataset_name:
            register_cityscapes_part(name=dataset_name,
                                     images_dirname=cfg.CUSTOM_DATASETS.CITYSCAPES_PART.IMAGES_DIRNAME,
                                     annotations_dirname=cfg.CUSTOM_DATASETS.CITYSCAPES_PART.ANNOTATIONS_DIRNAME,
                                     split=dataset_name.split('_')[-1],
                                     for_segmentation=True,
                                     path_only=cfg.CUSTOM_DATASETS.CITYSCAPES_PART.PATH_ONLY,
                                     debug=cfg.CUSTOM_DATASETS.CITYSCAPES_PART.DEBUG,
                                    )
            
        elif "pascal" in dataset_name:
            register_pascal_parts(
                name=dataset_name,
                images_dirname=cfg.CUSTOM_DATASETS.PASCAL_PARTS.IMAGES_DIRNAME,
                annotations_dirname=cfg.CUSTOM_DATASETS.PASCAL_PARTS.ANNOTATIONS_DIRNAME,
                split=dataset_name.split('_')[-1],
                year=2012, # Fixed.
                for_segmentation=True,
                subset_class_names=cfg.CUSTOM_DATASETS.PASCAL_PARTS.SUBSET_CLASS_NAMES,
                debug=cfg.CUSTOM_DATASETS.PASCAL_PARTS.DEBUG,
                )

        elif "imagenet" in dataset_name:
            register_imagenet_with_proposals(dataset_name,
                                             cfg.PART_RANKING.DATASET_PATH,
                                             "train",
                                             cfg.PART_RANKING.MIN_OBJECT_AREA_RATIO,
                                             partitioned_imagenet=bool(cfg.PART_RANKING.TOTAL_PARTITIONS > 0),
                                             total_partitions=cfg.PART_RANKING.TOTAL_PARTITIONS, 
                                             partition_index=cfg.PART_RANKING.PARTITION_INDEX,
                                             dataset_path_list=cfg.PART_RANKING.DATASET_PATH_LIST,
                                             filtered_code_path_list=cfg.PART_RANKING.FILTERED_CODE_PATH_LIST,
                                             exclude_code_path=cfg.PART_RANKING.EXCLUDE_CODE_PATH,
                                             path_only=cfg.PART_RANKING.PATH_ONLY,
                                             debug=cfg.PART_RANKING.DEBUG,
                                            )

    return cfg


def main(args):
    cfg = setup(args)
    if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
        run_name = cfg.WANDB.RUN_NAME 
        wandb.init(project=cfg.WANDB.PROJECT, sync_tensorboard=True, name=run_name,
         group=cfg.WANDB.GROUP, config=cfg.PART_RANKING, dir=cfg.OUTPUT_DIR)

    assert args.eval_only, "part ranking is eval-only."
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = Trainer.test(cfg, model)
    if comm.is_main_process():
        verify_results(cfg, res)
    if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
        wandb.finish()
    return res

   

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
