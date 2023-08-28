# Prepare models for PartDistillation

For training PartDistillation, we use [Mask2Former pretrained on COCO dataset for instance segmentation task](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md), and [Detic trained on ImageNet-21K, LVIS v1, and COCO datasets for open-vocabulary instance segmentation](https://github.com/facebookresearch/Detic/blob/main/docs/MODEL_ZOO.md). Specifically, we use the following pre-trained weights:
- Mask2Former: [[config](https://github.com/facebookresearch/Mask2Former/blob/main/configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml)] [[weight](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl)]
- Detic: [[config](https://github.com/facebookresearch/Detic/blob/main/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml)] [[weight](https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth)]

Please place the weights as following
```
$PART_DISTILLATION_ROOT/weights/
    m2f/instance/
        swinL_i21k_q200_e100.pkl 
    detic/
        Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```
*NOTE: Change the file names as above.*

