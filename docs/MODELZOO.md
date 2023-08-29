# PartDistillation Model Zoo

## Part Proposal Learning 
Part proposal learning models are trained following here: [[1K training](TRAINING_1K.md)], [[21K training (coming soon)]()]

To use pre-trained model for inference:
```
python part_proposal_train_net.py --config-file configs/PartProposalLearning.yaml --num-gpus 8 --num-machines 1 --eval-only \
PROPOSAL_LEARNING.MIN_OBJECT_AREA_RATIO 0.0 \
PROPOSAL_LEARNING.MIN_AREA_RATIO 0.0 \
MODEL.WEIGHTS /path/to/model/weights/name.pth \
OUTPUT_DIR /path/to/output/ 
```
for a part proposal model. 
Change `DATASETS.TEST` and `PROPOSAL_LEARNING.POSTPROCESS_TYPES` to evaluate `"prop"` (overlapping part proposals) or `"semseg"` (non-overlapping proposals). If W&B is setup, set `WANDB.DISABLE_WANDB` to `False` to visualize the predictions. See [here](TRAINING_1K.md) for available datasets.

To evaluate a supervised model:
```
python supervised_train_net.py --config-file configs/SupervisedPartProposalLearning.yaml --num-gpus 8 --num-machines 1 --eval-only \
DATASETS.TEST '("$DATASET_NAME",)' \
SUPERVISED_MODEL.USE_PER_PIXEL_LABEL True \
MODEL.WEIGHTS /path/to/model/weight/name.pth 
```
Change `DATASET_NAME` for different dataset evaluation and `SUPERVISED_MODEL.USE_PER_PIXEL_LABEL` to evaluate `"prop"` (overlapping part proposals) or `"semseg"` (non-overlapping proposals). If W&B is setup, set `WANDB.DISABLE_WANDB` to `False` to visualize the predictions. See [here](TRAINING_1K.md) for available datasets.

#### Evaluating Part-proposal Model for mIOU
Below will run clustering for each object-class first and use the cluster centroids as classifiers. 
```
python part_ranking_train_net.py --config-file configs/PartRanking.yaml --num-gpus 8 --num-machines 1 --eval-only \
DATASETS.TEST '("${DATASET_NAME}_pre_labeling_val","${DATASET_NAME}_match_val","${DATASET_NAME}_evaluate_val",)'
```
- *Change `DATASET_NAME` for different dataset. See [here](TRAINING_1K.md) for available datasets.*


### ImageNet-1K training 
|         config        | prediction type |  Pascal Part AR@200 | PartImageNet AR@200  | Download |
|-------------------|:---:|:------:|:-----------------:|:----------:|
|[PartDistillation (first-stage)](../configs/PartProposalLearning.yaml)  | overlapping | 27.2   | 52.2  | [model](https://utexas.box.com/shared/static/ovqrzxm9jwe66l0zjqyofkowk5zvhex1.pth) |
|[PartDistillation (first-stage)](../configs/PartProposalLearning.yaml)  | non-overlapping | 14.7  | 30.3 | [model](https://utexas.box.com/shared/static/ovqrzxm9jwe66l0zjqyofkowk5zvhex1.pth) |


## PartDistillation 

Final PartDistillation models are trained following here:[[1K training](TRAINING_1K.md)], [[21K training (coming soon)]()]. 

To evaluate the pre-trained PartDistillation model:
```
python part_distillation_train_net.py --config-file configs/PartDistillation.yaml --num-gpus 8 --num-machines 1 --eval-only \
PART_DISTILLATION.MIN_OBJECT_AREA_RATIO 0.0 \
PART_DISTILLATION.MIN_AREA_RATIO 0.0 \
MODEL.WEIGHTS /path/to/model/weights/name.pth \
OUTPUT_DIR /path/to/output/ 
```
Above commands runs inference for PartDistillation. 
Again, change `DATASET_NAME` for evaluating different datasets and `PROPOSAL_LEARNING.POSTPROCESS_TYPES` to evaluate `"prop"` (overlapping part proposals) or `"semseg"` (non-overlapping proposals). If W&B is setup, set `WANDB.DISABLE_WANDB` to `False` to visualize the predictions. See [here](TRAINING_1K.md) for available datasets.

To evaluate a supervised model:
```
python supervised_train_net.py --config-file configs/SupervisedLearning.yaml --num-gpus 8 --num-machines 1 --eval-only \
DATASETS.TEST '("$DATASET_NAME",)' \
SUPERVISED_MODEL.USE_PER_PIXEL_LABEL True \
MODEL.WEIGHTS /path/to/model/weight/name.pth 
```
Change `DATASETS.TEST` for different datasets (`pascal_part_val`, `part_imagenet_valtest`, etc.) See [here](TRAINING_1K.md) for available datasets.
If W&B is setup, set `WANDB.DISABLE_WANDB` to `False` to visualize the predictions. 


### ImageNet-1K training 
|         config        |  Pascal Part mIOU | PartImageNet mIOU  | Download |
|-----------------------|:--------------:|:-----------:|:-----------------:|
|[PartDistillation (second-stage)](../configs/PartDistillation.yaml)   |   22.3 | 46.0 | [model](https://utexas.box.com/shared/static/7651zj8n9ou3rbsmgfhjqobh7voxdnll.pth) |
