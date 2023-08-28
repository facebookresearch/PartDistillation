# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

batch_size=32
aug_list='["crop","scale","flip"]'
freeze_keys='["backbone","encoder"]'
LR="0.0001"
MIN_RATIO='0.001'
MIN_OBJECT_RATIO='0.001'
MIN_SCORE='-1.0'
MAX_ITER='80000'
NORM="True"
PER_PIXEL="True"

train_dataset='("imagenet_22k_train",)'
val_dataset='("pascal_match_val","pascal_evaluate_val",)'
# val_dataset='("part_imagenet_match_val","part_imagenet_evaluate_val",)'
# val_dataset='("cityscapes_part_match_val","cityscapes_part_evaluate_val",)'

oversample_ratio=3.0
inverse_sampling=False
importance_sampling_ratio=0.0

exp_name="Part_Distillation_Train"
num_obj_classes=22000
num_part_classes=8
pseudo_ann_path="pseudo_labels_saved/part_labels/part_masks_with_class/imagenet_22k_train/"
# grp_name="1k_detic_prediction"
grp_name="debug"
comment="lr_${LR}"

python3 "part_distillation_train_net.py" \
--config-file configs/part_distillation/swinL_IN21K_384_mask2former.yaml \
--num-gpus 8 \
--num-machines 1 \
OUTPUT_DIR "output/${exp_name}/${grp_name}/${comment}/" \
VIS_OUTPUT_DIR "vis_logs/${exp_name}/${grp_name}/${comment}/" \
DATASETS.TRAIN ${train_dataset} \
DATASETS.TEST ${val_dataset} \
MODEL.MASK_FORMER.QUERY_FEATURE_NORMALIZE ${NORM} \
MODEL.MASK_FORMER.FREEZE_KEYS ${freeze_keys} \
MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO  ${importance_sampling_ratio} \
MODEL.MASK_FORMER.OVERSAMPLE_RATIO ${oversample_ratio} \
WANDB.DISABLE_WANDB False \
WANDB.RUN_NAME ${comment} \
WANDB.PROJECT ${exp_name} \
WANDB.GROUP ${grp_name} \
WANDB.VIS_PERIOD_TRAIN 20 \
WANDB.VIS_PERIOD_TEST 20 \
SOLVER.MAX_ITER ${MAX_ITER} \
SOLVER.IMS_PER_BATCH ${batch_size} \
SOLVER.BASE_LR ${LR} \
SOLVER.STEPS '(70000, 75000)' \
CUSTOM_DATASETS.BASE_SIZE 640 \
CUSTOM_DATASETS.AUG_NAME_LIST ${aug_list} \
PART_DISTILLATION.MIN_OBJECT_AREA_RATIO ${MIN_OBJECT_RATIO} \
PART_DISTILLATION.MIN_AREA_RATIO ${MIN_RATIO} \
PART_DISTILLATION.MIN_SCORE ${MIN_SCORE} \
PART_DISTILLATION.DATASET_PATH ${pseudo_ann_path} \
PART_DISTILLATION.PATH_ONLY True \
PART_DISTILLATION.NUM_OBJECT_CLASSES ${num_obj_classes} \
PART_DISTILLATION.NUM_PART_CLASSES ${num_part_classes} \
PART_DISTILLATION.USE_PER_PIXEL_LABEL ${PER_PIXEL} \
PART_DISTILLATION.SET_IMAGE_SQUARE True \
PART_DISTILLATION.DEBUG True
