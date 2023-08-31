# PartDistillation: Learning Parts from Instance Segmentation 

PartDistillation learns to segment parts over 21k object categories without labels.

<p align="center"> <img src='teaser.png' align="center" height="200px"> </p>

> [**PartDistillation: Learning Parts from Instance Segmentation**](http://arxiv.org/abs/xxxx),            
> Jang Hyun Cho, Philipp Kr&auml;henb&uuml;hl, Vignash Ramanathan,
> *arxiv ([arXiv xxxx](http://arxiv.org/abs/xxxx))*  

Contact: janghyuncho7@utexas.edu

## Installation 
Please see [installation instructions](). 

*Internal: See `INSTALL.md`.*

## Getting Started

See instructions for [preparing datasets]() and [preparing models]() for PartDistillation.

*Internal: See `datasets/README.md` and `weights/README.md`.*

## Training PartDistillation
PartDistillation has multiple stages to train the full model. 
Parts are separated with object segmentation and we use Detic predictions to do the job. 
To make the process fast, we save all detic predictions of ImageNet first. 

### Save detic prediction

```
./sh_files/detic/run.sh 
``` 
Above code will launch 60 parallel jobs to run detic and save the result at `pseudo_labels/object_labels/imagenet_22k_train/detic_predictions/`.


### Pixel grouping for class-agnostic part segments


```
./sh_files/proposal_generation/run.sh
```

Above code will launch 40 parallel jobs. Pixel-grouping is good initial segments, but rough.
Need to smooth out with postprocessing. We postprocess all part segments with dense-CRF with the following command.
 
```
./sh_files/dcrf/run.sh 
``` 
*NOTE: change the number of processes in the submit files to accommodate the resource availability.*


Then, we start training part proposal model (1st stage), which is a class-agnostic part segmentation model based on Mask2Former. 


### Part-proposal Learning

```
./sh_files/proposal_learning/train_multi.sh 
```

Above code will train on 4 nodes with 256 batch size. Then, we need to establish global association for each object class.
This allows to produce consistent class label for each part during inference. We call this process *part ranking*.

### Part Ranking

```
./sh_files/part_ranking/run.sh
```

This generate part segmentation labels with class (as cluster assignment). With this, we self-train 
the entire system all-together.

### PartDistillation Training

```
./sh_files/part_distillation_training/train.sh 
```

This will launch 4 node job training on entire ImageNet-21K dataset. 


## Ablations

### Supervised learning 

Supervised models can be trained with commands in `sh_files/supervised_learning/`. For example,

```
./sh_files/supervised_learning/semseg/pascal.sh 
```
will launch a 4 node job training a Mask2Former-based model (same configuration as ours) on `train` split of Pascal Parts dataset.

### Fewshot training 

In, `sh_files/fewshot_learning/` there are all training commands for training fewshot. For example,

```
./sh_files/supervised_learning/semseg/pascal.sh 
```
will launch 1 node job finetuning a pretrained model in fewshot setting. 

*NOTE: please modify `model_weights` and `percent` variables based on your needs.*


# Locations (Internal)

## Original codes 

```
/private/home/janghyuncho7/EmergentPartSeg
```

## Notebooks 

```
/private/home/janghyuncho7/EmergentPartSeg/notebooks/
```

## Refactored code 

``` 
/private/home/janghyuncho7/PartDistillation
``` 

## Collages 
``` 
/checkpoint/janghyuncho7/PartDistillation/manual_eval_related/collages/
```
- diversity evaluation: `/checkpoint/janghyuncho7/PartDistillation/manual_eval_related/diversity_eval`
- collages (imagenet-22k): `/checkpoint/janghyuncho7/PartDistillation/manual_eval_related/collages/imagenet_22k_train/detic_predictions/`
- one-stage baseline: `/checkpoint/janghyuncho7/PartDistillation/manual_eval_related/collages/imagenet_1k_train` 
- per-pixel baseline: `/checkpoint/janghyuncho7/PartDistillation/manual_eval_related/collages/imagenet_22k_train/per_pixel_learning_baseline/collage_3x3/`


## Pseudo labels 

```
/checkpoint/janghyuncho7/PartDistillation/pseudo_labels_saved/
``` 
- detic prediction: `/checkpoint/janghyuncho7/PartDistillation/pseudo_labels_saved/object_labels/imagenet_22k_train/detic_predictions/`.
- part segments by pixel grouping (IN22K, detic): `/checkpoint/janghyuncho7/PartDistillation/pseudo_labels_saved/part_labels/proposal_generation/imagenet_22k_train/detic_based/generated_proposals_new_processed/res3_res4/dot_4_norm_False/`
- part segments by pixel grouping (IN1K, m2f COCO): `/checkpoint/janghyuncho7/PartDistillation/pseudo_labels_saved/part_labels/proposal_generation/imagenet_1k_train/generated_proposals_processed/score_based/res4/l2_4/` 
- part segmentation labels by part ranking (IN22K): `/checkpoint/janghyuncho7/PartDistillation/pseudo_labels_saved/part_labels/part_masks_with_class/imagenet_22k_train/`

## trained models 

``` 
/checkpoint/janghyuncho7/PartDistillation/models/
```
- initial pretrain weights: `/checkpoint/janghyuncho7/PartDistillation/models/pre_weights/weights`
- per-pixel baselines: `/checkpoint/janghyuncho7/PartDistillation/models/per_pixel_baselines`
- our models: `/checkpoint/janghyuncho7/PartDistillation/models/our_models`
- few-shot models: `/checkpoint/janghyuncho7/PartDistillation/models/fewshot`

## everything else

```
/checkpoint/janghyuncho7/PartDistillation/
```

## Reproduction

New annotations and models are saved in `pseudo_labels/` and `output/`. Please change the path in submit files if new annotations and models will be used in public.  

## License
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
