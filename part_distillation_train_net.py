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

import sys 
import os 
import torch
import torch.nn as nn 
import numpy as np 
import logging
import detectron2.utils.comm as comm
import wandb 

sys.path.append('Detic/third_party/CenterNet2')
sys.path.append('Detic/third_party/Deformable-DETR')

from collections import OrderedDict
from pathlib import Path

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog, 
                             build_detection_test_loader, 
                             build_detection_train_loader)

from detectron2.engine import (default_argument_parser,
                               default_setup,
                               launch)

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import is_main_process, synchronize
from detectron2.evaluation import verify_results, inference_on_dataset, print_csv_format

from part_distillation import (add_maskformer2_config, 
                               add_wandb_config, 
                               add_custom_datasets_config,
                               add_part_distillation_config)

from part_distillation.data.datasets.register_pascal_parts import register_pascal_parts
from part_distillation.data.datasets.register_cityscapes_part import register_cityscapes_part
from part_distillation.data.datasets.register_part_imagenet import register_part_imagenet
from part_distillation.data.datasets.register_imagenet_with_proposals import register_imagenet_with_proposals
from part_distillation.data.datasets.register_imagenet_with_segmentation import register_imagenet_with_segmentation
from part_distillation.data.datasets.register_imagenet import register_imagenet

from part_distillation.data.dataset_mappers.part_distillation_dataset_mapper import PartDistillationDatasetMapper
from part_distillation.data.dataset_mappers.proposal_dataset_mapper import ProposalDatasetMapper
from part_distillation.data.dataset_mappers.voc_parts_mapper import VOCPartsMapper
from part_distillation.data.dataset_mappers.cityscapes_part_mapper import CityscapesPartMapper
from part_distillation.data.dataset_mappers.part_imagenet_mapper import PartImageNetMapper

from part_distillation.evaluation.miou_evaluator import mIOU_Evaluator
from part_distillation.evaluation.miou_matcher import mIOU_Matcher
 
from base_trainer import BaseTrainer, maybe_dp, get_mode



class Trainer(BaseTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        if "match" in dataset_name:
            return mIOU_Matcher(dataset_name, 
                                num_classes=cfg.PART_DISTILLATION.NUM_PART_CLASSES)
        elif "evaluate" in dataset_name:
            return mIOU_Evaluator(dataset_name)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if "pascal" in dataset_name:
            mapper = VOCPartsMapper(cfg, is_train=False)
        elif "part_imagenet" in dataset_name:
            mapper = PartImageNetMapper(cfg, dataset_name, is_train=False)
        elif "cityscapes" in dataset_name:
            mapper = CityscapesPartMapper(cfg, is_train=False)
        elif "save_labels" in dataset_name:
            mapper = PartDistillationDatasetMapper(cfg, is_train=False)

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


    @classmethod
    def build_train_loader(cls, cfg):
        mapper = PartDistillationDatasetMapper(cfg, is_train=True)

        return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def test(cls, cfg, model):
        logger  = logging.getLogger("part_distillation")
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            mode = get_mode(dataset_name)
            logger.info("Starting {} mode on {}.".format(mode, dataset_name))
            maybe_dp(model).register_metadata(dataset_name)
            maybe_dp(model).mode = mode 

            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = cls.build_evaluator(cfg, dataset_name)
            results_i = inference_on_dataset(model, data_loader, evaluator)
            
            if mode == "match":
                maybe_dp(model).update_majority_vote_mapping(results_i)
                logger.info("Majority vote result:\n{}".format(results_i))
                continue 
            maybe_dp(model).register_metadata(cfg.DATASETS.TRAIN[0]) # reset to training dataset.
            maybe_dp(model).mode = ""                                # reset mode. 
  
            results.update(results_i)
            if comm.is_main_process():
                assert isinstance(results_i, dict), \
                "Evaluator must return a dict on the main process. Got {} instead.".format(results_i)
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
            comm.synchronize()

            # add dataset name
            results = {dataset_name + "_" + k: v for k, v in results.items()}

        if len(results) == 1:
            results = list(results.values())[0]
        
        comm.synchronize()
        if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
            wandb.log(results)

        return results
        

def setup(args):
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_part_distillation_config(cfg)
    add_custom_datasets_config(cfg)
    add_wandb_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="part_distillation")
    
    # for part-imagenet mapping.
    register_imagenet("imagenet_1k_meta_train", "train",
                      partitioned_imagenet=False)

    # register dataset
    register_imagenet_with_segmentation(cfg.DATASETS.TRAIN[0],
                                        cfg.PART_DISTILLATION.DATASET_PATH,
                                        "train",
                                        dataset_path_list=cfg.PART_DISTILLATION.DATASET_PATH_LIST,
                                        filtered_code_path_list=cfg.PART_DISTILLATION.FILTERED_CODE_PATH_LIST,
                                        exclude_code_path=cfg.PART_DISTILLATION.EXCLUDE_CODE_PATH,
                                        path_only=cfg.PART_DISTILLATION.PATH_ONLY,
                                        debug=cfg.PART_DISTILLATION.DEBUG,
                                        )

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
                subset_class_names=cfg.CUSTOM_DATASETS.PASCAL_PARTS.SUBSET_CLASS_NAMES,
                for_segmentation=True,
                debug=cfg.CUSTOM_DATASETS.PASCAL_PARTS.DEBUG,
                )
        elif "imagenet" in dataset_name:
            register_imagenet_with_segmentation(dataset_name,
                                                cfg.PART_DISTILLATION.DATASET_PATH,
                                                "train",
                                                partitioned_imagenet=bool(cfg.PART_DISTILLATION.TOTAL_PARTITIONS > 0),
                                                total_partitions=cfg.PART_DISTILLATION.TOTAL_PARTITIONS, 
                                                partition_index=cfg.PART_DISTILLATION.PARTITION_INDEX,
                                                dataset_path_list=cfg.PART_DISTILLATION.DATASET_PATH_LIST,
                                                filtered_code_path_list=cfg.PART_DISTILLATION.FILTERED_CODE_PATH_LIST,
                                                exclude_code_path=cfg.PART_DISTILLATION.EXCLUDE_CODE_PATH,
                                                path_only=cfg.PART_DISTILLATION.PATH_ONLY,
                                                debug=cfg.PART_DISTILLATION.DEBUG,
                                                )
        else:
            raise ValueError("{} not supported.".format(dataset_name))

    return cfg


def main(args):
    cfg = setup(args)
    if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
        run_name = cfg.WANDB.RUN_NAME 
        wandb.init(project=cfg.WANDB.PROJECT, sync_tensorboard=True, name=run_name,
         group=cfg.WANDB.GROUP, config=cfg.PART_DISTILLATION, dir=cfg.OUTPUT_DIR)

    if args.eval_only:
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
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    res = trainer.train()
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
