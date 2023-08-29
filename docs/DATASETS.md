# Prepare datasets for PartDistillation

For training PartDistillation, we use [ImageNet-22K](https://www.image-net.org/download.php) for original setup and [ImageNet-1K](https://www.image-net.org/download.php) ([huggingface](https://huggingface.co/datasets/imagenet-1k)) for compute-friendly setup. We evaluate our models on [Pascal Parts](http://roozbehm.info/pascal-parts/pascal-parts.html), [PartImageNet](https://github.com/TACJu/PartImageNet), and [Cityscapes Part](https://github.com/mcordts/cityscapesScripts) datasets. Please download these datasets from the official websites and place or sim-link under `$PART_DISTILLATION_ROOT/datasets/`. 

```
$PART_DISTILLATION_ROOT/datasets/
    imagenet_1k/
    imagenet_22k/
    part_imagenet/
    pascal_parts/
    cityscapes_part/ 
```


## ImageNet-1K
For compute-friendly setting, one can train PartDistillation with ImageNet-1K dataset. Please download dataset and place them as following

```
imagenet_1k/
    train/
        n01440764
        n01443537
        ...
    val/
        n01440764
        n01443537
        ...
```


## ImageNet-21K 
Download dataset and place them as following
```
imagenet_21k/
    synsets.dat 
    words.txt 
    ...
    n02090622/
        n02090622_10.JPEG
        n02090622_100.JPEG
        ...
    ...

```


## PartImageNet 
Please download PartImageNet from the [original source](https://github.com/TACJu/PartImageNet) and place them as following

```
part_imagenet/
    train.json 
    val.json
    test.json 
    valtest.json # copy from datasets/metadata/part_imagenet_valtest.json
    train/
    val/
        n01484850
        ...
    test/
        n01491361
        ...
    valtest/
        n01484850 # from val/
        ...
        n01491361 # from test/
        ...
```

*NOTE: `valtest` and `valtest.json` are not provided by the original source. We simply combined `val.json` and `test.json` and provided in `datasets/metadata/`. Please make a new directory `valtest` and simply copy or sim-link folders inside `valtest`, and copy `valtest.json` from `datasets/metadata/part_imagenet_valtest.json`.*

## Pascal Parts
Pascal Parts dataset uses [the images of Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) and the annotations of [Pascal Parts](http://roozbehm.info/pascal-parts/pascal-parts.html). Please download them and place as following

```
pascal_parts/
    images/  # from Pascal VOC 2012
        ImageSets/
        JPEGImages/
        ...
    annotations/ # from Pascal Parts
        2008_000002.mat
        2008_000003.mat
        2008_000007.mat
        ...
```


## Cityscapes Part 
Cityscapes Part is a panoptic part segmentation dataset but we use instance only. Please download the dataset from [the official source](https://github.com/pmeletis/panoptic_parts) and place as following

```
cityscapes_part/
    leftImage8bit/
    gtFinePanopticParts/
        train/
            aachen/
            ...
        val/
            frankfurt/
            ...
```
