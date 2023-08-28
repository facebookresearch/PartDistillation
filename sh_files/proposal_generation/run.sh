# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

NUM_CLUSTERS=4
DEBUG_MODE=False
DATASET_NAME=imagenet_22k
SPLIT=train
metric="dot"
feat_list='["res3","res4"]'
N_IMS=1
feat_norm=False
TOT_IDS=40
exp_name="ProposalGeneration"
grp_name="res34_${metric}"
object_mask_path="pseudo_labels_saved/object_labels/imagenet_22k_train/detic_predictions/"
for ID in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40
do
    comment="k${NUM_CLUSTERS}i${ID}"

    python3 "multi_node_train_net.py" \
    --config-file configs/proposal_generation/swinL_IN21K_384_mask2former.yaml \
    --num-gpus 8 -p "learnaccel" \
    --num-machines 1 \
    --name ${comment} \
    --target "proposal_generation_net.py" \
    --job-dir "output/${exp_name}/${grp_name}/${comment}/" \
    VIS_OUTPUT_DIR "vis_logs/${exp_name}/${grp_name}/${comment}/" \
    OUTPUT_DIR "output/${exp_name}/${grp_name}/${comment}/" \
    WANDB.DISABLE_WANDB False \
    WANDB.RUN_NAME ${comment} \
    WANDB.PROJECT ${exp_name} \
    WANDB.GROUP ${grp_name} \
    WANDB.VIS_PERIOD_TEST 2000 \
    PROPOSAL_GENERATION.OBJECT_MASK_PATH ${object_mask_path} \
    DATASETS.TEST "('${DATASET_NAME}"_"${SPLIT}',)" \
    MODEL.META_ARCHITECTURE "ProposalGenerationModel" \
    PROPOSAL_GENERATION.NUM_SUPERPIXEL_CLUSTERS ${NUM_CLUSTERS} \
    PROPOSAL_GENERATION.DATASET_NAME ${DATASET_NAME}"_"${SPLIT} \
    PROPOSAL_GENERATION.PARTITION_INDEX ${ID} \
    PROPOSAL_GENERATION.TOTAL_PARTITIONS ${TOT_IDS} \
    PROPOSAL_GENERATION.OBJECT_MASK_TYPE "detic" \
    PROPOSAL_GENERATION.WITH_GIVEN_MASK True \
    PROPOSAL_GENERATION.DISTANCE_METRIC ${metric} \
    PROPOSAL_GENERATION.BACKBONE_FEATURE_KEY_LIST ${feat_list} \
    PROPOSAL_GENERATION.FEATURE_NORMALIZE ${feat_norm} \
    PROPOSAL_GENERATION.BATCH_SIZE ${N_IMS} \
    PROPOSAL_GENERATION.DEBUG ${DEBUG_MODE}
done
