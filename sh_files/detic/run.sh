# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DEBUG_MODE=False
DATASET_NAME=imagenet_22k
SPLIT=train
N_IMS=2
TOT_IDS=60
for ID in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60
do
    python3 "multi_node_train_net.py" \
    --config-file configs/detic/Detic_Labeling.yaml \
    --num-gpus 8 -p "learnaccel" \
    --num-machines 1 \
    --eval-only \
    --name "detic_labeling_${ID}" \
    --target "detic_labeling_net.py" \
    --job-dir "output/detic_22k/detic_labeling_${ID}/" \
    OUTPUT_DIR "output/detic_22k/" \
    PROPOSAL_GENERATION.BATCH_SIZE ${N_IMS} \
    PROPOSAL_GENERATION.DEBUG ${DEBUG_MODE} \
    PROPOSAL_GENERATION.DATASET_NAME ${DATASET_NAME}"_"${SPLIT} \
    PROPOSAL_GENERATION.PARTITION_INDEX ${ID} \
    PROPOSAL_GENERATION.TOTAL_PARTITIONS ${TOT_IDS} \
    INPUT.IMAGE_SIZE 640 \
    DATASETS.TEST "('${DATASET_NAME}_${SPLIT}',)" \
    TEST.DETECTIONS_PER_IMAGE 1000
done


# DEBUG_MODE=True
# DATASET_NAME=imagenet_1k
# SPLIT=train
# N_IMS=2
# ID=10
# TOT_IDS=10
# python3 "detic_labeling_net.py" \
# --config-file configs/detic/Detic_Labeling.yaml \
# --num-gpus 2 \
# --num-machines 1 \
# --eval-only \
# OUTPUT_DIR "output/detic/" \
# PROPOSAL_GENERATION.BATCH_SIZE ${N_IMS} \
# PROPOSAL_GENERATION.DEBUG ${DEBUG_MODE} \
# PROPOSAL_GENERATION.DATASET_NAME ${DATASET_NAME}"_"${SPLIT} \
# PROPOSAL_GENERATION.PARALLEL_JOB_ID ${ID} \
# PROPOSAL_GENERATION.NUM_PARALLEL_JOBS ${TOT_IDS} \
# INPUT.IMAGE_SIZE 640 \
# TEST.DETECTIONS_PER_IMAGE 1000
