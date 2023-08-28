# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

batch_size=32
train_dataset='("imagenet_22k_train",)'
val_dataset='("imagenet_22k_pre_labeling_train","imagenet_22k_post_labeling_train",)'


# Control factors
PER_PIXEL_CLUSTERING="True"
PER_PIXEL_LABELING="True"
MIN_RATIO1='0.05'
MIN_SCORE1='0.3'
MIN_RATIO2='0.05'
MIN_SCORE2='0.1'
num_pcluster=8
cls_metric="l2"
prop_key="decoder_output"
FEAT_NORM=True
partition=True
# pid=0
total_p=50


LR="0.0001"
model_id="0049999"

exp_name="Part_Ranking"
grp_name="IN21K+COCO"

# Chnage this
prop_model_type="detic_and_score"
prop_model_name="lr_${LR}"
weight_path=""
obj_mask_type="detic_predictions"
pseudo_ann_path="pseudo_labels/part_labels/imagenet_22k_train/${obj_mask_type}/"

for pid in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
do
    comment="pp_${PER_PIXEL_CLUSTERING}_s_${MIN_SCORE1}_${pid}"

    python3 "multi_node_train_net.py" \
    --config-file configs/part_ranking/swinL_IN21K_384_mask2former.yaml \
    --num-gpus 8 -p "learnaccel" \
    --num-machines 1 \
    --eval-only \
    --target "part_ranking_train_net.py" \
    --job-dir "output/${exp_name}/${grp_name}/${comment}/" \
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
    PART_RANKING.USE_PER_PIXEL_LABEL_DURING_CLUSTERING ${PER_PIXEL_CLUSTERING} \
    PART_RANKING.USE_PER_PIXEL_LABEL_DURING_LABELING ${PER_PIXEL_LABELING} \
    PART_RANKING.PROPOSAL_KEY ${prop_key} \
    PART_RANKING.CLASSIFIER_METRIC ${cls_metric} \
    PART_RANKING.NUM_CLUSTERS ${num_pcluster} \
    PART_RANKING.APPLY_MASKING_WITH_OBJECT_MASK True \
    PART_RANKING.OBJECT_MASK_TYPE ${obj_mask_type} \
    PART_RANKING.PROPOSAL_MODEL_TYPE ${prop_model_type} \
    PART_RANKING.PROPOSAL_MODEL_NAME ${prop_model_name} \
    PART_RANKING.PROPOSAL_FEATURE_NORM ${FEAT_NORM} \
    PART_RANKING.PARTITION_INDEX ${pid} \
    PART_RANKING.TOTAL_PARTITIONS ${total_p} \
    PART_RANKING.DEBUG False
done
