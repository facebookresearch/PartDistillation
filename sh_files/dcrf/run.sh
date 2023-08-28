# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

TOT_IDS=90
for ID in 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90
do
    python3 continuously_postprocess_dcrf.py \
    --parallel_job_id $ID \
    --num_parallel_jobs $TOT_IDS \
    --res "res4" \
    --num_k 4 &
done
