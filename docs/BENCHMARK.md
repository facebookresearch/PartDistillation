# Benchmark Training and Evaluation

In our experiments, we compare our method to supervised baselines as well as our *one-stage* baseline. 

### Supervised baseline training 
In our experiments, we train supervised baseline for (1) proposal model and (2) part segmentation model. 

#### Supervised Part-proposal Model
Use the following command to train a supervised part proposal model baseline:

```
python supervised_train_net.py --config-file configs/SupervisedPartProposalLearning.yaml --num-gpus 8 --num-machines 1 \
DATASETS.TRAIN '("$DATASET_NAME",)' 
```
- *Change `DATASET_NAME` for training on different datasets.*
- *Change `SUPERVISED_MODEL.USE_PER_PIXEL_LABEL` to `True` for evaluating non-overlapping proposals.* 

#### Supervised Part Segmentation Model
Use the following command to train a supervised part segmentation model baseline:

```
python supervised_train_net.py --config-file configs/SupervisedLearning.yaml --num-gpus 8 --num-machines 1 \
DATASETS.TRAIN '("$DATASET_NAME",)' 
```
- *Change `DATASET_NAME` for training on different datasets.*

### Fewshot training 

We simply initialize Mask2Former with PartDistillation and train with `$PERCENT` amount of human annotations. Similar to before, we train for *Part Proposals* and *Part Segmentation*. To train for few-shot experiments, use the following command:

#### Part-proposal Model
```
python supervised_train_net.py --config-file configs/SupervisedPartProposalLearning.yaml --num-gpus 8 --num-machines 1 \
MODEL.WEIGHTS /path/to/pretrained/weights/name.pth \
FEWSHOT_LEARNING.LABEL_PERCENTAGE $PERCENT \
DATASETS.TRAIN '("${TRAINSET}",)' \
DATASETS.TRAIN '("${TESTSET}",)' 
```
- *NOTE: Change `TRAINSET` and `TESTSET` for training on different datasets. Change `PERCENT` for different % of human labels.*

#### Part Segmentation Model
```
python supervised_train_net.py --config-file configs/SupervisedLearning.yaml --num-gpus 8 --num-machines 1 \
MODEL.WEIGHTS /path/to/pretrained/weights/name.pth \
FEWSHOT_LEARNING.LABEL_PERCENTAGE $PERCENT \
DATASETS.TRAIN '("${TRAINSET}",)' \
DATASETS.TRAIN '("${TESTSET}",)' 
```
- *NOTE: Change `TRAINSET` and `TESTSET` for training on different datasets. Change `PERCENT` for different % of human labels.*

