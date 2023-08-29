# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List
from detectron2.utils.comm import synchronize
from detectron2.evaluation.evaluator import DatasetEvaluator


class NullEvaluator(DatasetEvaluator):
    def reset(self):
        return 

    def process(self, inputs: List[Dict], outputs: Dict):
        return 
            
    def evaluate(self):
        synchronize()
        return