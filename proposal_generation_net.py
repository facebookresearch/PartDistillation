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
import itertools
import logging
import os
import sys
import torch
import detectron2.utils.comm as comm
import wandb

sys.path.append('Detic/third_party/CenterNet2')
sys.path.append('Detic/third_party/Deformable-DETR')

from collections import OrderedDict
from typing import Any, Dict, List, Set
from pathlib import Path


from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog,
                             build_detection_test_loader)

from detectron2.engine import (DefaultTrainer,
                               default_argument_parser,
                               default_setup,
                               launch)

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from detectron2.utils.comm import is_main_process, synchronize
from detectron2.evaluation import verify_results
from part_distillation import add_maskformer2_config, add_proposal_generation_config, add_wandb_config
from part_distillation.data.dataset_mappers.proposal_generation_mapper import ProposalGenerationMapper
from part_distillation.evaluation.null_evaluator import NullEvaluator
from part_distillation.data.datasets.register_imagenet import register_imagenet


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(self, *args, **kwargs):
        return NullEvaluator()


    @classmethod
    def build_test_loader(self, cfg, dataset_name):
        mapper = ProposalGenerationMapper(cfg)
        return build_detection_test_loader(cfg, dataset_name,
                                           batch_size=cfg.PROPOSAL_GENERATION.BATCH_SIZE,
                                           mapper=mapper)


    @classmethod
    def test(self, cfg, model, evaluators=None):
        results = super().test(cfg, model, evaluators)
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
    add_proposal_generation_config(cfg)
    add_wandb_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="part_distillation")
    dataset_name_dir = cfg.PROPOSAL_GENERATION.DATASET_NAME if not cfg.PROPOSAL_GENERATION.DEBUG else "debug"
    save_path = "pseudo_labels/part_labels/proposal_generation/{}/{}/{}/{}_{}_norm_{}/"\
                .format(dataset_name_dir,
                cfg.PROPOSAL_GENERATION.OBJECT_MASK_TYPE,
                "_".join(cfg.PROPOSAL_GENERATION.BACKBONE_FEATURE_KEY_LIST),
                cfg.PROPOSAL_GENERATION.DISTANCE_METRIC,
                cfg.PROPOSAL_GENERATION.NUM_SUPERPIXEL_CLUSTERS,
                cfg.PROPOSAL_GENERATION.FEATURE_NORMALIZE)

    # register dataset
    register_imagenet(
        cfg.PROPOSAL_GENERATION.DATASET_NAME,
        split=cfg.PROPOSAL_GENERATION.DATASET_NAME.split('_')[-1],
        partitioned_imagenet=bool(cfg.PROPOSAL_GENERATION.TOTAL_PARTITIONS > 0),
        partition_index=cfg.PROPOSAL_GENERATION.PARTITION_INDEX,
        total_partitions=cfg.PROPOSAL_GENERATION.TOTAL_PARTITIONS,
        save_path=save_path,
        with_given_mask=cfg.PROPOSAL_GENERATION.WITH_GIVEN_MASK,
        filtered_code_path_list=cfg.PROPOSAL_GENERATION.FILTERED_CODE_PATH_LIST,
        object_mask_path=cfg.PROPOSAL_GENERATION.OBJECT_MASK_PATH,
        exclude_code_path=cfg.PROPOSAL_GENERATION.EXCLUDE_CODE_PATH,
        single_class_code=cfg.PROPOSAL_GENERATION.SINGLE_CLASS_CODE,
        use_part_imagenet_classes=cfg.PROPOSAL_GENERATION.USE_PART_IMAGENET_CLASSES,
        debug=cfg.PROPOSAL_GENERATION.DEBUG,
        )

    return cfg



def main(args):
    cfg = setup(args)
    if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
        run_name = Path(cfg.OUTPUT_DIR).name
        wandb.init(project=cfg.WANDB.PROJECT, sync_tensorboard=True, name=run_name,
         group=cfg.WANDB.GROUP, config=cfg.PROPOSAL_GENERATION, dir=cfg.OUTPUT_DIR)

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
