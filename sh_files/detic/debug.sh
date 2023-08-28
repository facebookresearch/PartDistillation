# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DEBUG_MODE=True
DATASET_NAME=imagenet_1k
SPLIT=train
N_IMS=2
ID=1
TOT_IDS=50
python3 "detic_labeling_net.py" \
--config-file configs/detic/Detic_Labeling.yaml \
--num-gpus 1 \
--num-machines 1 \
--eval-only \
OUTPUT_DIR "output/detic_22k/" \
PROPOSAL_GENERATION.BATCH_SIZE ${N_IMS} \
PROPOSAL_GENERATION.DEBUG ${DEBUG_MODE} \
PROPOSAL_GENERATION.DATASET_NAME ${DATASET_NAME}"_"${SPLIT} \
PROPOSAL_GENERATION.PARTITION_INDEX ${ID} \
PROPOSAL_GENERATION.TOTAL_PARTITIONS ${TOT_IDS} \
INPUT.IMAGE_SIZE 640 \
DATASETS.TEST "('${DATASET_NAME}_${SPLIT}',)" \
TEST.DETECTIONS_PER_IMAGE 1000
