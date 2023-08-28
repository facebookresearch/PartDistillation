# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

train_dataset='("imagenet_1k_train",)'
val_dataset='("part_imagenet_valtest",)'
exp_name="PixelGrouping_Evaluation"

feat_norm=False
metric="dot"
grp_name="swinL_m2f"
comment="debug"
NUM_CLUSTERS=4
feat_list='["res3","res4"]'


python3 "pixel_grouping_test_net.py" \
--config-file configs/pixel_grouping/swinL_IN21K_384_mask2former.yaml \
--num-gpus 8 \
--num-machines 1 \
--eval-only \
OUTPUT_DIR "output/${exp_name}/${grp_name}/${comment}/" \
VIS_OUTPUT_DIR "vis_logs/${exp_name}/${grp_name}/${comment}/" \
DATASETS.TEST ${val_dataset} \
WANDB.RUN_NAME ${comment} \
WANDB.PROJECT ${exp_name} \
WANDB.GROUP ${grp_name} \
WANDB.DISABLE_WANDB False \
WANDB.VIS_PERIOD_TEST 30 \
CUSTOM_DATASETS.USE_MERGED_GT True \
PIXEL_GROUPING.NUM_SUPERPIXEL_CLUSTERS ${NUM_CLUSTERS} \
PIXEL_GROUPING.DISTANCE_METRIC ${metric} \
PIXEL_GROUPING.BACKBONE_FEATURE_KEY_LIST ${feat_list} \
TEST.EVAL_PERIOD 50 \
CUSTOM_DATASETS.PART_IMAGENET.DEBUG False
