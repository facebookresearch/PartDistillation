
# Training PartDistillation

PartDistillation has multiple stages to train the full model. 
Parts are separated with object segmentation and we use Detic predictions to do the job. 
To make the process fast, we save all detic predictions of ImageNet first. 

### Save detic prediction
First, we need to download pretrained detic weight. Download it [here](https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth) and place it in `weights/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth`.
Then, use the pretrained Detic model to precompute object instance segmentation:
```
./sh_files/detic/run.sh 
``` 
Above code will launch 60 parallel jobs to run detic and save the result at `pseudo_labels/object_labels/imagenet_22k_train/detic_predictions/`.


### Pixel grouping for class-agnostic part segments
Please donwload pretrained mask2former weight [here](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl) and place it in `weights/mask2former/instance/swinL_i21k_q200_e100.pkl`.

```
./sh_files/proposal_generation/run.sh
```

Above code will launch 40 parallel jobs. Pixel-grouping is good initial segments, but rough.
Need to smooth out with postprocessing. We postprocess all part segments with dense-CRF with the following command.
 
```
./sh_files/dcrf/run.sh 
``` 
*NOTE: change the number of processes in the submit files to accomodate the resource availability.*

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
