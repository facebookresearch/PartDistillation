# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

batch_size=512
aug_list='["crop","scale","flip"]'
freeze_keys='["backbone","encoder"]'
LR="0.0001"
MIN_OBJ_RATIO='0.05'
MIN_RATIO='0.05'
MIN_SCORE='-1.0'
MAX_ITER='50000'
PER_PIXEL="True"

train_dataset='("imagenet_22k_train",)'
val_dataset='("pascal_part_val","part_imagenet_valtest","cityscapes_part_val",)'
process_list='("prop","prop","prop",)'
oversample_ratio=3.0
inverse_sampling=False
importance_sampling_ratio=0.0

exp_name="Proposal_Learning_Train"
grp_name="IN22K+COCO"

pseudo_ann_path="pseudo_labels_saved/part_labels/proposal_generation/imagenet_22k_train/detic_based/generated_proposals_new_processed/res3_res4/dot_4_norm_False/"
pseudo_ann_path_extra="pseudo_labels_saved/part_labels/proposal_generation/imagenet_1k_train/generated_proposals_processed/score_based/res4/l2_4/"
filtered_code_path_list='()'
comment="lr_${LR}"

python3 "multi_node_train_net.py" \
--config-file configs/proposal_learning/swinL_IN21K_384_mask2former.yaml \
--num-gpus 8 -p "learnaccel" \
--num-machines 8 \
--resume \
--use-volta32 \
--name "prop" \
--target "part_proposal_train_net.py" \
--job-dir "output/${exp_name}/${grp_name}/${comment}/multi_node/" \
OUTPUT_DIR "output/${exp_name}/${grp_name}/${comment}/" \
VIS_OUTPUT_DIR "vis_logs/${exp_name}/${grp_name}/${comment}/" \
DATASETS.TRAIN ${train_dataset} \
DATASETS.TEST ${val_dataset} \
CUSTOM_DATASETS.AUG_NAME_LIST ${aug_list} \
CUSTOM_DATASETS.BASE_SIZE 640 \
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
SOLVER.STEPS '(40000, 45000)' \
TEST.EVAL_PERIOD 10000 \
PROPOSAL_LEARNING.MIN_OBJECT_AREA_RATIO ${MIN_OBJ_RATIO} \
PROPOSAL_LEARNING.MIN_AREA_RATIO ${MIN_RATIO} \
PROPOSAL_LEARNING.MIN_SCORE ${MIN_SCORE} \
CUSTOM_DATASETS.USE_MERGED_GT True \
PROPOSAL_LEARNING.DATASET_PATH_LIST "('${pseudo_ann_path}','${pseudo_ann_path_extra}',)" \
PROPOSAL_LEARNING.DATASET_PATH ${pseudo_ann_path} \
PROPOSAL_LEARNING.USE_PER_PIXEL_LABEL ${PER_PIXEL} \
PROPOSAL_LEARNING.APPLY_MASKING_WITH_OBJECT_MASK True \
PROPOSAL_LEARNING.FILTERED_CODE_PATH_LIST ${filtered_code_path_list} \
PROPOSAL_LEARNING.POSTPROCESS_TYPES ${process_list} \
PROPOSAL_LEARNING.PATH_ONLY True \
PROPOSAL_LEARNING.DEBUG False
