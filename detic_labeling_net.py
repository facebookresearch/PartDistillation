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

import logging
import os
import sys
import torch
import detectron2.utils.comm as comm

from collections import OrderedDict
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog, 
                             build_detection_test_loader)

from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (inference_on_dataset, print_csv_format)
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

sys.path.append('Detic/third_party/CenterNet2')
from centernet.config import add_centernet_config

sys.path.append('Detic/third_party/Deformable-DETR')
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test


from part_distillation.data.dataset_mappers.proposal_generation_mapper import ProposalGenerationMapper
from part_distillation.evaluation.null_evaluator import NullEvaluator
from part_distillation import add_maskformer2_config, add_proposal_generation_config, add_wandb_config
from part_distillation.data.datasets.register_imagenet import register_imagenet
from base_trainer import maybe_dp

logger = logging.getLogger("detectron2")


def get_clip_embeddings(vocabulary, prompt='a '):
    from Detic.detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb



def prepare_model(model, dataset_name, labeling_mode, score_thres, debug):
    logger.info("preparing model for {}.".format(dataset_name))
    metadata = MetadataCatalog.get(dataset_name)
    maybe_dp(model).register_metadata(metadata, labeling_mode, score_thres, debug)

    # Setup clip classifier with class names.
    if labeling_mode == 'human-only':
        metadata.class_names = ["person", "man", "woman", "toddler", "human"] 
    else:
        metadata.class_names = metadata.classes
    classifier = get_clip_embeddings(metadata.class_names)
    num_classes = len(metadata.class_names)
    reset_cls_test(model, classifier, num_classes)

    return model



def do_label(cfg, model):
    results = OrderedDict()
    for d, dataset_name in enumerate(cfg.DATASETS.TEST):
        label_mode = cfg.PROPOSAL_GENERATION.DETIC_LABELING_MODE
        score_thres = cfg.PROPOSAL_GENERATION.SAVE_SCORE_THRESHOLD
        model  = prepare_model(model, dataset_name, label_mode, score_thres, 
                               cfg.PROPOSAL_GENERATION.DEBUG)
        mapper = ProposalGenerationMapper(cfg)
        data_loader = build_detection_test_loader(cfg, dataset_name, 
                                                  batch_size=cfg.PROPOSAL_GENERATION.BATCH_SIZE, 
                                                  mapper=mapper)
        evaluator = NullEvaluator()
        results[dataset_name] = inference_on_dataset(model, data_loader, evaluator)
        
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    
    return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_maskformer2_config(cfg)
    add_detic_config(cfg)
    add_proposal_generation_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="part_distillation")

    dataset_name_dir = cfg.PROPOSAL_GENERATION.DATASET_NAME if not cfg.PROPOSAL_GENERATION.DEBUG else "debug"
    detic_labeling_mode = cfg.PROPOSAL_GENERATION.DETIC_LABELING_MODE
    root_folder_name = cfg.PROPOSAL_GENERATION.ROOT_FOLDER_NAME
    save_path = f"{root_folder_name}/object_labels/detic_predictions/{detic_labeling_mode}/{dataset_name_dir}/"
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

    metadata = MetadataCatalog.get(cfg.PROPOSAL_GENERATION.DATASET_NAME)
    if comm.is_main_process():
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for fname in metadata.class_codes:
            folder_path = os.path.join(save_path, fname)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
    comm.synchronize()

    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    assert args.eval_only, "detic-labeling is eval-only."
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )

    return do_label(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser()
    args = args.parse_args()
    if args.num_machines == 1:
        args.dist_url = 'tcp://127.0.0.1:{}'.format(
            torch.randint(11111, 60000, (1,))[0].item())
    else:
        if args.dist_url == 'host':
            args.dist_url = 'tcp://{}:12345'.format(
                os.environ['SLURM_JOB_NODELIST'])
        elif not args.dist_url.startswith('tcp'):
            tmp = os.popen(
                    'echo $(scontrol show job {} | grep BatchHost)'.format(
                        args.dist_url)
                ).read()
            tmp = tmp[tmp.find('=') + 1: -1]
            args.dist_url = 'tcp://{}:12345'.format(tmp)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
