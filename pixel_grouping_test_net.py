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
import torch
import detectron2.utils.comm as comm
import wandb

sys.path.append('Detic/third_party/CenterNet2')
sys.path.append('Detic/third_party/Deformable-DETR')

from collections import OrderedDict
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
from detectron2.evaluation import verify_results, inference_on_dataset, print_csv_format

from part_distillation import (add_maskformer2_config,
                               add_wandb_config,
                               add_pixel_grouping_confing,
                               add_custom_datasets_config)

from part_distillation.data.datasets.register_imagenet import register_imagenet
from part_distillation.data.datasets.register_part_imagenet import register_part_imagenet
from part_distillation.data.dataset_mappers.part_imagenet_mapper import PartImageNetMapper
from part_distillation.evaluation.proposal_evaluator import ProposalEvaluator

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(self, *args, **kwargs):
        return ProposalEvaluator()

    @classmethod
    def build_test_loader(self, cfg, dataset_name):
        mapper = PartImageNetMapper(cfg, is_train=False)

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


    @classmethod
    def test(cls, cfg, model):
        logger  = logging.getLogger("part_distillation")
        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            evaluator = cls.build_evaluator(cfg, dataset_name)
            results_i = inference_on_dataset(model, data_loader, evaluator)

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
    add_pixel_grouping_confing(cfg)
    add_custom_datasets_config(cfg)
    add_wandb_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    # Setup logger
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="part_distillation")
    # To use the metadata
    register_imagenet("imagenet_1k_meta_train", "train",
                      partitioned_imagenet=False)
    for dataset_name in cfg.DATASETS.TEST:
        if "part_imagenet" in dataset_name:
            register_part_imagenet(name=dataset_name,
                                   images_dirname=cfg.CUSTOM_DATASETS.PART_IMAGENET.IMAGES_DIRNAME,
                                   annotations_dirname=cfg.CUSTOM_DATASETS.PART_IMAGENET.ANNOTATIONS_DIRNAME,
                                   split=dataset_name.split('_')[-1],
                                   debug=cfg.CUSTOM_DATASETS.PART_IMAGENET.DEBUG,
                                  )
        else:
            raise ValueError("{} not supported for pixel grouping evaluation.".format(dataset_name))

    return cfg


def main(args):
    cfg = setup(args)
    if comm.is_main_process() and not cfg.WANDB.DISABLE_WANDB:
        run_name = cfg.WANDB.RUN_NAME
        wandb.init(project=cfg.WANDB.PROJECT, sync_tensorboard=True, name=run_name,
         group=cfg.WANDB.GROUP, config=cfg.PIXEL_GROUPING, dir=cfg.VIS_OUTPUT_DIR)

    assert args.eval_only, "pixel grouping is eval-only."
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
