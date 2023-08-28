# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

batch_size=32
train_dataset='("imagenet_1k_train",)'
val_dataset='("imagenet_1k_pre_labeling_train","imagenet_1k_post_labeling_train",)'


# Control factors
PER_PIXEL_CLUSTERING="True"
PER_PIXEL_LABELING="True"
MIN_RATIO1='0.05'
MIN_SCORE1='0.3'
MIN_RATIO2='0.01'
MIN_SCORE2='0.1'
num_pcluster=8
cls_metric="l2"
prop_key="decoder_output"
FEAT_NORM=True

model_id="0049999"

exp_name="Part_Ranking"
grp_name="IN21K+COCO"

# Chnage this
prop_model_type="detic_and_score"
prop_model_name="lr_${LR}"
weight_path="weights/proposal_model/lr_0.0001_0039999.pth"
pseudo_ann_path="pseudo_labels_saved/part_labels/proposal_generation/imagenet_22k_train/detic_based/generated_proposals_new_processed/res3_res4/dot_4_norm_False/"
pseudo_ann_path_extra="pseudo_labels_saved/part_labels/proposal_generation/imagenet_1k_train/generated_proposals_processed/score_based/res4/l2_4/"
pid=2
total_p=50
comment="pp_${PER_PIXEL_CLUSTERING}_s_${MIN_SCORE1}_${pid}"

python3 "part_ranking_train_net.py" \
--config-file configs/part_ranking/swinL_IN21K_384_mask2former.yaml \
--num-gpus 8 \
--num-machines 1 \
--eval-only \
MODEL.WEIGHTS  ${weight_path} \
OUTPUT_DIR "output/${exp_name}/${grp_name}/${comment}/" \
DATASETS.TRAIN ${train_dataset} \
DATASETS.TEST ${val_dataset} \
WANDB.DISABLE_WANDB False \
WANDB.RUN_NAME ${comment} \
WANDB.PROJECT ${exp_name} \
WANDB.GROUP ${grp_name} \
WANDB.VIS_PERIOD_TRAIN 200 \
WANDB.VIS_PERIOD_TEST 20 \
PART_RANKING.MIN_AREA_RATIO_1 ${MIN_RATIO1} \
PART_RANKING.MIN_SCORE_1 ${MIN_SCORE1} \
PART_RANKING.MIN_AREA_RATIO_2 ${MIN_RATIO2} \
PART_RANKING.MIN_SCORE_2 ${MIN_SCORE2} \
CUSTOM_DATASETS.USE_MERGED_GT True \
PART_RANKING.DATASET_PATH ${pseudo_ann_path} \
PART_RANKING.DATASET_PATH_LIST "('${pseudo_ann_path}','${pseudo_ann_path_extra}',)" \
PART_RANKING.USE_PER_PIXEL_LABEL_DURING_CLUSTERING ${PER_PIXEL_CLUSTERING} \
PART_RANKING.USE_PER_PIXEL_LABEL_DURING_LABELING ${PER_PIXEL_LABELING} \
PART_RANKING.PROPOSAL_KEY ${prop_key} \
PART_RANKING.CLASSIFIER_METRIC ${cls_metric} \
PART_RANKING.NUM_CLUSTERS ${num_pcluster} \
PART_RANKING.APPLY_MASKING_WITH_OBJECT_MASK True \
PART_RANKING.PROPOSAL_FEATURE_NORM ${FEAT_NORM} \
PART_RANKING.PARTITION_INDEX ${pid} \
PART_RANKING.TOTAL_PARTITIONS ${total_p} \
PART_RANKING.DEBUG True
