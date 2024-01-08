# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    cfg.INPUT.IMAGE_SIZE_BASE = 640

    # solver config 
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112 
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_MATCH = 112 * 112
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS_LOSS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # NOTE: Added config for PartDistillation.
    cfg.MODEL.MASK_FORMER.FREEZE_KEYS = []
    cfg.MODEL.MASK_FORMER.QUERY_FEATURE_NORMALIZE = False 

    # fp16 
    cfg.FP16 = False 
    cfg.USE_CHECKPOINT = False 


def add_wandb_config(cfg):
    cfg.WANDB = CN() 
    cfg.WANDB.DISABLE_WANDB = False 
    cfg.WANDB.GROUP = None 
    cfg.WANDB.PROJECT = ""
    cfg.WANDB.VIS_PERIOD_TRAIN = 200
    cfg.WANDB.VIS_PERIOD_TEST = 20
    cfg.WANDB.RUN_NAME = "output"
    cfg.DATASETS.DEBUG = False
    cfg.WANDB.VIS_TOPK = 10
    cfg.VIS_OUTPUT_DIR = ""



def add_proposal_learning_config(cfg):
    cfg.PROPOSAL_LEARNING = CN() 
    cfg.PROPOSAL_LEARNING.MIN_OBJECT_AREA_RATIO = 0.001 
    cfg.PROPOSAL_LEARNING.MIN_AREA_RATIO = 0.0
    cfg.PROPOSAL_LEARNING.MIN_SCORE = -1.0
    cfg.PROPOSAL_LEARNING.DATASET_PATH_LIST = []
    cfg.PROPOSAL_LEARNING.FILTERED_CODE_PATH_LIST = []
    cfg.PROPOSAL_LEARNING.EXCLUDE_CODE_PATH = ""
    cfg.PROPOSAL_LEARNING.PATH_ONLY = False 
    cfg.PROPOSAL_LEARNING.USE_PER_PIXEL_LABEL = True
    cfg.PROPOSAL_LEARNING.DATASET_PATH = ""
    cfg.PROPOSAL_LEARNING.LABEL_PERCENTAGE = 100 
    cfg.PROPOSAL_LEARNING.APPLY_MASKING_WITH_OBJECT_MASK = True  
    cfg.PROPOSAL_LEARNING.POSTPROCESS_TYPES = []
    cfg.PROPOSAL_LEARNING.DEBUG = False 




def add_custom_datasets_config(cfg):
    cfg.CUSTOM_DATASETS = CN()
    cfg.CUSTOM_DATASETS.BASE_SIZE = -1
    cfg.CUSTOM_DATASETS.AUG_NAME_LIST = []
    cfg.CUSTOM_DATASETS.USE_MERGED_GT = True 
    cfg.CUSTOM_DATASETS.LABEL_PERCENTAGE = 100

    cfg.CUSTOM_DATASETS.PASCAL_PARTS = CN()
    cfg.CUSTOM_DATASETS.PASCAL_PARTS.IMAGES_DIRNAME = ""
    cfg.CUSTOM_DATASETS.PASCAL_PARTS.ANNOTATIONS_DIRNAME = ""
    cfg.CUSTOM_DATASETS.PASCAL_PARTS.SUBSET_CLASS_NAMES = []
    cfg.CUSTOM_DATASETS.PASCAL_PARTS.DEBUG = False 

    cfg.CUSTOM_DATASETS.CITYSCAPES_PART = CN()
    cfg.CUSTOM_DATASETS.CITYSCAPES_PART.IMAGES_DIRNAME = "" 
    cfg.CUSTOM_DATASETS.CITYSCAPES_PART.ANNOTATIONS_DIRNAME = "" 
    cfg.CUSTOM_DATASETS.CITYSCAPES_PART.PATH_ONLY = False 
    cfg.CUSTOM_DATASETS.CITYSCAPES_PART.DEBUG = False 

    cfg.CUSTOM_DATASETS.PART_IMAGENET = CN()
    cfg.CUSTOM_DATASETS.PART_IMAGENET.IMAGES_DIRNAME = "" 
    cfg.CUSTOM_DATASETS.PART_IMAGENET.ANNOTATIONS_DIRNAME = "" 
    cfg.CUSTOM_DATASETS.PART_IMAGENET.DEBUG = False 
    



def add_proposal_generation_config(cfg):
    cfg.PROPOSAL_GENERATION = CN() 
    cfg.PROPOSAL_GENERATION.DATASET_NAME = "imagenet_22k_train"
    cfg.PROPOSAL_GENERATION.OBJECT_MASK_TYPE = "detic"
    cfg.PROPOSAL_GENERATION.OBJECT_MASK_PATH = "pseudo_labels/object_labels/imagenet_22k_train/detic_predictions/"
    cfg.PROPOSAL_GENERATION.NUM_SUPERPIXEL_CLUSTERS = 4 
    cfg.PROPOSAL_GENERATION.DISTANCE_METRIC = "l2"
    cfg.PROPOSAL_GENERATION.FEATURE_NORMALIZE = False
    cfg.PROPOSAL_GENERATION.BACKBONE_FEATURE_KEY_LIST = ["res4"]
    cfg.PROPOSAL_GENERATION.TOTAL_PARTITIONS = -1
    cfg.PROPOSAL_GENERATION.PARTITION_INDEX = -1 
    cfg.PROPOSAL_GENERATION.BATCH_SIZE = 4
    cfg.PROPOSAL_GENERATION.WITH_GIVEN_MASK = False 
    cfg.PROPOSAL_GENERATION.USE_PART_IMAGENET_CLASSES = False 
    cfg.PROPOSAL_GENERATION.FILTERED_CODE_PATH_LIST = []
    cfg.PROPOSAL_GENERATION.EXCLUDE_CODE_PATH = ""
    cfg.PROPOSAL_GENERATION.SINGLE_CLASS_CODE = ""
    cfg.PROPOSAL_GENERATION.ROOT_FOLDER_NAME = "pseudo_labels"
    cfg.PROPOSAL_GENERATION.DETIC_LABELING_MODE = "max-gt-label" # "max-gt-label" or "human-only"
    cfg.PROPOSAL_GENERATION.SAVE_SCORE_THRESHOLD = 0.0
    cfg.PROPOSAL_GENERATION.DEBUG = False 
    

    
def add_part_ranking_config(cfg):
    cfg.PART_RANKING = CN() 
    cfg.PART_RANKING.DATASET_PATH = ""
    cfg.PART_RANKING.DATASET_PATH_LIST = []
    cfg.PART_RANKING.FILTERED_CODE_PATH_LIST = []
    cfg.PART_RANKING.EXCLUDE_CODE_PATH = ""
    cfg.PART_RANKING.PATH_ONLY = False 
    cfg.PART_RANKING.NUM_CLUSTERS = 8
    cfg.PART_RANKING.CLASSIFIER_METRIC = "l2"
    cfg.PART_RANKING.PROPOSAL_KEY = "decoder_output"
    cfg.PART_RANKING.PROPOSAL_FEATURE_NORM = True 
    cfg.PART_RANKING.MIN_OBJECT_AREA_RATIO = 0.001 
    cfg.PART_RANKING.MIN_AREA_RATIO_1 = 0.0
    cfg.PART_RANKING.MIN_AREA_RATIO_2 = 0.0
    cfg.PART_RANKING.MIN_SCORE_1 = 0.0
    cfg.PART_RANKING.MIN_SCORE_2 = 0.0
    cfg.PART_RANKING.USE_PER_PIXEL_LABEL_DURING_CLUSTERING = True 
    cfg.PART_RANKING.USE_PER_PIXEL_LABEL_DURING_LABELING = True
    cfg.PART_RANKING.APPLY_MASKING_WITH_OBJECT_MASK = True 
    cfg.PART_RANKING.TOTAL_PARTITIONS = -1
    cfg.PART_RANKING.PARTITION_INDEX = -1 
    cfg.PART_RANKING.ROOT_FOLDER_NAME = "pseudo_labels"
    cfg.PART_RANKING.WEIGHT_NAME = "default"
    cfg.PART_RANKING.SAVE_ANNOTATIONS = False 
    cfg.PART_RANKING.DEBUG = False 


def add_part_distillation_config(cfg):
    cfg.PART_DISTILLATION = CN() 
    cfg.PART_DISTILLATION.DATASET_PATH = ""
    cfg.PART_DISTILLATION.DATASET_PATH_LIST = []
    cfg.PART_DISTILLATION.FILTERED_CODE_PATH_LIST = []
    cfg.PART_DISTILLATION.EXCLUDE_CODE_PATH = ""
    cfg.PART_DISTILLATION.PATH_ONLY = False 
    cfg.PART_DISTILLATION.USE_PER_PIXEL_LABEL = True
    cfg.PART_DISTILLATION.NUM_PART_CLASSES = 8
    cfg.PART_DISTILLATION.NUM_OBJECT_CLASSES = 1000 # ImageNet-1K
    cfg.PART_DISTILLATION.MIN_OBJECT_AREA_RATIO = 0.001 
    cfg.PART_DISTILLATION.MIN_AREA_RATIO = -1.0
    cfg.PART_DISTILLATION.MIN_SCORE = -1.0
    cfg.PART_DISTILLATION.USE_ORACLE_CLASSIFIER = False 
    cfg.PART_DISTILLATION.APPLY_MASKING_WITH_OBJECT_MASK = True 
    cfg.PART_DISTILLATION.TOTAL_PARTITIONS = -1
    cfg.PART_DISTILLATION.PARTITION_INDEX = -1 
    cfg.PART_DISTILLATION.SET_IMAGE_SQUARE = False 
    cfg.PART_DISTILLATION.DEBUG = False 



def add_pixel_grouping_confing(cfg):
    cfg.PIXEL_GROUPING = CN()
    cfg.PIXEL_GROUPING.NUM_SUPERPIXEL_CLUSTERS = 4 
    cfg.PIXEL_GROUPING.DISTANCE_METRIC = "l2"
    cfg.PIXEL_GROUPING.BACKBONE_FEATURE_KEY_LIST = ["res4"]
    cfg.PIXEL_GROUPING.FEATURE_NORMALIZE = False
    cfg.PIXEL_GROUPING.DEBUG = False




def add_supervised_model_config(cfg):
    cfg.SUPERVISED_MODEL = CN() 
    cfg.SUPERVISED_MODEL.USE_PER_PIXEL_LABEL = False 
    cfg.SUPERVISED_MODEL.APPLY_MASKING_WITH_OBJECT_MASK = True 
    cfg.SUPERVISED_MODEL.CLASS_AGNOSTIC_LEARNING = False 
    cfg.SUPERVISED_MODEL.CLASS_AGNOSTIC_INFERENCE = False 


def add_fewshot_learning_config(cfg):
    cfg.FEWSHOT_LEARNING = CN() 
    cfg.FEWSHOT_LEARNING.LABEL_PERCENTAGE = 100