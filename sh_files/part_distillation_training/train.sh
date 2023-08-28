# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

batch_size=256
aug_list='["crop","scale","flip"]'
freeze_keys='["backbone","encoder"]'
LR="0.0001"
MIN_RATIO='0.05'
MIN_OBJECT_RATIO='0.05'
MIN_SCORE='-1.0'
MAX_ITER='120000'
PER_PIXEL="True"

train_dataset='("imagenet_22k_train",)'
val_dataset='("part_imagenet_match_valtest","part_imagenet_evaluate_valtest",)'

oversample_ratio=3.0
importance_sampling_ratio=0.0

exp_name="Part_Distillation_Train"
num_obj_classes=22000
num_part_classes=8
pseudo_ann_path="pseudo_labels_saved/part_labels/part_masks_with_class/imagenet_22k_train/"
grp_name="IN22K"
comment="lr_${LR}"

python3 "multi_node_train_net.py" \
--config-file configs/part_distillation/swinL_IN21K_384_mask2former.yaml \
--num-gpus 8 -p "learnaccel" \
--num-machines 8 \
--use-volta32 \
--name ${comment} \
--target "part_distillation_train_net.py" \
--job-dir "output/${exp_name}/${grp_name}/${comment}/multi_node/" \
OUTPUT_DIR "output/${exp_name}/${grp_name}/${comment}/" \
VIS_OUTPUT_DIR "vis_logs/${exp_name}/${grp_name}/${comment}/" \
DATASETS.TRAIN ${train_dataset} \
DATASETS.TEST ${val_dataset} \
MODEL.MASK_FORMER.FREEZE_KEYS ${freeze_keys} \
MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO  ${importance_sampling_ratio} \
MODEL.MASK_FORMER.OVERSAMPLE_RATIO ${oversample_ratio} \
WANDB.DISABLE_WANDB False \
WANDB.RUN_NAME ${comment} \
WANDB.PROJECT ${exp_name} \
WANDB.GROUP ${grp_name} \
WANDB.VIS_PERIOD_TRAIN 1000 \
WANDB.VIS_PERIOD_TEST 50 \
SOLVER.MAX_ITER ${MAX_ITER} \
SOLVER.IMS_PER_BATCH ${batch_size} \
SOLVER.BASE_LR ${LR} \
SOLVER.STEPS '(100000, 110000)' \
CUSTOM_DATASETS.BASE_SIZE 640 \
CUSTOM_DATASETS.AUG_NAME_LIST ${aug_list} \
CUSTOM_DATASETS.MIN_OBJECT_AREA_RATIO ${MIN_OBJECT_RATIO} \
CUSTOM_DATASETS.MIN_AREA_RATIO ${MIN_RATIO} \
CUSTOM_DATASETS.MIN_SCORE ${MIN_SCORE} \
CUSTOM_DATASETS.DATASET_PATH ${pseudo_ann_path} \
CUSTOM_DATASETS.PATH_ONLY True \
PART_DISTILLATION.NUM_OBJECT_CLASSES ${num_obj_classes} \
PART_DISTILLATION.NUM_PART_CLASSES ${num_part_classes} \
PART_DISTILLATION.USE_PER_PIXEL_LABEL ${PER_PIXEL} \
PART_DISTILLATION.SET_IMAGE_SQUARE True \
CUSTOM_DATASETS.DEBUG False
