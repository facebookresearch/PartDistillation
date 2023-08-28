# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

batch_size=32
aug_list='["crop","scale","flip"]'
freeze_keys='[]'
LR="0.0001"
MAX_ITER='20000'
NORM="True"
PER_PIXEL="True"

train_dataset='("cityscapes_part_train",)'
val_dataset='('
val_dataset=${val_dataset}'"cityscapes_part_val",'
val_dataset=${val_dataset}')'

oversample_ratio=3.0
inverse_sampling=False
importance_sampling_ratio=0.75

exp_name="Semseg_Fewshot_Learning"
grp_name="Cityscapes_Part"

model_weights=""  # change this to the trained model path.
percent=100       # change this to desired percentage.
comment="coco_m2f_${percent}" # if model_weight is empty, it will initialize m2f coco-instance segmentation.

python3 "multi_node_train_net.py" \
--config-file configs/supervised_train_net/swinL_IN21K_384_mask2former.yaml \
--num-gpus 8 -p "learnaccel" \
--num-machines 1 \
--name "cp_semseg_${percent}" \
--target "supervised_train_net.py" \
--job-dir "output/${exp_name}/${grp_name}/${comment}/multi_node/" \
MODEL.WEIGHTS ${model_weights} \
OUTPUT_DIR "output/${exp_name}/${grp_name}/${comment}/" \
VIS_OUTPUT_DIR "vis_logs/${exp_name}/${grp_name}/${comment}/" \
DATASETS.TRAIN ${train_dataset} \
DATASETS.TEST ${val_dataset} \
MODEL.MASK_FORMER.FREEZE_KEYS ${freeze_keys} \
MODEL.MASK_FORMER.QUERY_FEATURE_NORMALIZE ${NORM} \
MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO  ${importance_sampling_ratio} \
MODEL.MASK_FORMER.OVERSAMPLE_RATIO ${oversample_ratio} \
WANDB.DISABLE_WANDB False \
WANDB.RUN_NAME ${comment} \
WANDB.PROJECT ${exp_name} \
WANDB.GROUP ${grp_name} \
WANDB.VIS_PERIOD_TRAIN 200 \
WANDB.VIS_PERIOD_TEST 50 \
SOLVER.MAX_ITER ${MAX_ITER} \
SOLVER.IMS_PER_BATCH ${batch_size} \
SOLVER.BASE_LR ${LR} \
SOLVER.STEPS '(40000, 45000)' \
TEST.EVAL_PERIOD 20000 \
CUSTOM_DATASETS.USE_MERGED_GT True \
CUSTOM_DATASETS.AUG_NAME_LIST ${aug_list} \
MODEL.SEM_SEG_HEAD.NUM_CLASSES 50 \
SUPERVISED_MODEL.USE_PER_PIXEL_LABEL ${PER_PIXEL} \
SUPERVISED_MODEL.CLASS_AGNOSTIC_INFERENCE False \
SUPERVISED_MODEL.APPLY_MASKING_WITH_OBJECT_MASK True \
SUPERVISED_MODEL.CLASS_AGNOSTIC_LEARNING False \
FEWSHOT_LEARNING.LABEL_PERCENTAGE ${percent} \
CUSTOM_DATASETS.PASCAL_PARTS.DEBUG False
