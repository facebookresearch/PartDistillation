# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import copy
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from detectron2.data.detection_utils import read_image
from pycocotools import mask as coco_mask
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask
from detectron2.structures import Instances
from detectron2.data import transforms as T
from PIL import Image, ImageDraw, ImageFont


IMAGE_SIZE = 640

_RED = np.array([1.0, 0, 0])

class Partvisualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        colors = "r"
        alpha = 0.3

        self.overlay_instances(
            masks=masks,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


def ann_to_instance_dict(data):
    masks = torch.tensor([coco_mask.decode(ann["segmentation"]) for ann in data["part_masks"]])
    label = data["part_labels"]

    instance_dict = {}
    for msk, lbl in zip(masks, label):
        instance = Instances(masks.shape[1:])
        instance.pred_masks = msk[None]
        instance.pred_classes = lbl[None]

        instance_dict[lbl.item()] = instance
    return instance_dict




def make_collage(n, pathlist):
    assert n**2 == len(pathlist), "pathlist size needs to be {}.".format(n**2)
    collage = np.zeros((n*IMAGE_SIZE, n*IMAGE_SIZE, 3), dtype=np.uint8)
    for i, path in enumerate(pathlist):
        image = Image.open(path)
        d = ImageDraw.Draw(image)
        font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
        font = ImageFont.truetype(font_path, size=100)
        d.text((0, 0), "{}".format(i+1), font=font, fill=(0, 0, 0))
        image = np.array(image)
        h, w = image.shape[:2]

        a = (i//n)*IMAGE_SIZE
        b = (i%n )*IMAGE_SIZE
        collage[a:a+h, b:b+w] = image
    return Image.fromarray(collage)




def get_vis_image(data, instance, opacity=0.9):
    image = read_image(data["file_name"])
    image = T.apply_transform_gens(augs, image)[0]
    white = np.ones(image.shape) * 255
    image = image * opacity + white * (1-opacity)
    visualizer = Partvisualizer(image)
    vis_image = visualizer.draw_instance_predictions(predictions=instance).get_image()
    vis_image = Image.fromarray(vis_image)

    return vis_image


augs = [T.ResizeScale(min_scale=1.0, max_scale=1.0, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE),
        T.FixedSizeCrop(crop_size=(IMAGE_SIZE, IMAGE_SIZE))
        ]




def get_argparse():
    parser = argparse.ArgumentParser(description='Postprocess visualization')
    parser.add_argument('--parallel_job_id', type=int, default=-1)
    parser.add_argument('--num_parallel_jobs', type=int, default=-1)
    parser.add_argument('--num_parts', type=int, default=8)
    parser.add_argument('--model_name', type=str, default="lr_0.0001")
    parser.add_argument('--object_mask_type', type=str, default="detic_and_score")
    parser.add_argument('--mask_ranking_type', type=str, default="detic_predictions")
    parser.add_argument('--dataset_name', type=str, default="imagenet_22k_train")
    parser.add_argument('--pseudo_root_folder', type=str, default="pseudo_labels_saved")
    parser.add_argument('--comment', type=str, default="human_eval")
    parser.add_argument('--make_collage', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--image_limit', type=int, default=-1)
    parser.add_argument('--collage_limit', type=int, default=-1)
    parser.add_argument('--collage_size', type=int, default=3)
    parser.add_argument('--mode', type=str, default="clustered_proposal_labels")

    return parser.parse_args()


IMAGENET_22K_DATASET_PATH = "datasets/imagenet_22k/"

if __name__ == "__main__":
    args = get_argparse()

    with open(os.path.join(IMAGENET_22K_DATASET_PATH, "synsets.dat"), "r") as f:
        class_code_list = f.readlines()
    class_code_list = [_.strip() for _ in class_code_list]
    with open(os.path.join(IMAGENET_22K_DATASET_PATH, "words.txt"), "r") as f:
        fname_cname_pair_list = f.readlines()
    fname_to_classname = {x.split('\t')[0]: x.split('\t')[1].strip() for x in fname_cname_pair_list}
    fname_to_classname = {k:v for k, v in fname_to_classname.items() if k in class_code_list}

    # For clustered labels.
    if args.mode == "clustered_proposal_labels":
        source_root = f"{args.pseudo_root_folder}/part_labels/part_masks_with_class/{args.dataset_name}/{args.mask_ranking_type}/{args.object_mask_type}/{args.model_name}/local_l2_4/masking_step_24/global_l2_{args.num_parts}/"
        target_root = f"visualization/{args.dataset_name}/{args.mask_ranking_type}/{args.object_mask_type}/{args.model_name}/local_l2_4/masking_step_24/global_l2_{args.num_parts}/"
        collage_root = f"collages/{args.dataset_name}/{args.mask_ranking_type}/{args.object_mask_type}/{args.model_name}/local_l2_4/masking_step_24/global_l2_{args.num_parts}/"

    # For model predictions.
    if args.mode == "model_predictions":
        source_root = f"visualization/{args.dataset_name}/{args.model_name}/"
        target_root = f"visualization/{args.dataset_name}/overlayed_images/{args.model_name}/"
        collage_root = f"collages/{args.dataset_name}/{args.model_name}/"

    num_parts = args.num_parts

    if args.mode == "supervised":
        source_root = f"{args.pseudo_root_folder}/part_labels/part_masks_with_class/{args.dataset_name}/{args.mask_ranking_type}/{args.object_mask_type}/{args.model_name}/local_l2_4/masking_step_24/global_l2_{args.num_parts}/"
        target_root = f"visualization/{args.dataset_name}/{args.mask_ranking_type}/{args.object_mask_type}/{args.model_name}/{args.comment}/local_l2_4/masking_step_24/global_l2_{args.num_parts}/"
        collage_root = f"collages/{args.dataset_name}/{args.mask_ranking_type}/{args.object_mask_type}/{args.model_name}/{args.comment}/local_l2_4/masking_step_24/global_l2_{args.num_parts}/"
        code_list = torch.load(f"datasets/metadata/{args.comment}.pkl")
    else:
        code_list = os.listdir(source_root)

    if args.num_parallel_jobs > 0:
        num_total_classes = len(code_list)
        num_classes_per_job = num_total_classes // args.num_parallel_jobs
        num_remaining_classes = num_total_classes - args.num_parallel_jobs * num_classes_per_job
        num_current_job_classes = num_classes_per_job

        start_i = num_current_job_classes * (args.parallel_job_id-1)
        end_i = num_current_job_classes * args.parallel_job_id
        if args.parallel_job_id == args.num_parallel_jobs:
            end_i = num_total_classes
        code_list = code_list[start_i:end_i]

    if args.make_collage:
        if not args.debug:
            progress_count = 0
            for code in code_list:
                # folder_name = code
                progress_count += 1
                folder_name = code + "_" + fname_to_classname[code]
                pname_list = os.listdir(os.path.join(target_root, folder_name))
                for pname in pname_list:
                    pathlist = []
                    count = 0
                    collage_id = 0
                    for fname in os.listdir(os.path.join(target_root, folder_name, pname)):
                        pathlist.append(os.path.join(target_root, folder_name, pname, fname))
                        count += 1

                        if count % args.collage_size**2 == 0:
                            collage = make_collage(args.collage_size, pathlist)
                            if not os.path.exists(os.path.join(collage_root, "collage_{}x{}".format(args.collage_size, args.collage_size), folder_name, pname)):
                                os.makedirs(os.path.join(collage_root, "collage_{}x{}".format(args.collage_size, args.collage_size), folder_name, pname))
                            collage.save(os.path.join(collage_root, "collage_{}x{}".format(args.collage_size, args.collage_size), folder_name, pname, fname))
                            pathlist = []
                            collage_id += 1

                            if args.collage_limit > 0 and args.collage_limit < collage_id:
                                break
                if progress_count % 5 == 0:
                    print('{:.2f} \% done.'.format(progress_count/len(code_list) * 100), flush=True)
        else:
            pathlist = []
            count = 0
            collage_id = 0
            for fname in os.listdir("debug_vis"):
                pathlist.append(os.path.join("debug_vis", fname))
                count += 1

                if count % args.collage_size**2 == 0:
                    collage = make_collage(args.collage_size, pathlist)
                    collage.save("debug_collage/collage_{}.png".format(collage_id))
                    pathlist = []
                    collage_id += 1

        print("Done.")


    else:
        count = 0
        debug_count = 0
        for code in code_list:
            count += 1
            folder_name = code + "_" + fname_to_classname[code]
            fname_list = os.listdir(os.path.join(source_root, code))
            if args.image_limit > 0:
                fname_list = fname_list[:args.image_limit]
            for fname in fname_list:
                data = torch.load(os.path.join(source_root, code, fname), "cpu")

                instance_dict = ann_to_instance_dict(data)

                for part_id, instance in instance_dict.items():
                    if not args.debug and not os.path.exists(os.path.join(target_root, folder_name, "part_{}".format(part_id))):
                        os.makedirs(os.path.join(target_root, folder_name, "part_{}".format(part_id)))

                    if not os.path.exists(os.path.join(target_root, folder_name, "part_{}".format(part_id), fname)):
                        vis_image = get_vis_image(data, instance, 0.7)
                        debug_count += 1
                        if args.debug and debug_count <= 500:
                            print(vis_image, os.path.join(target_root, folder_name, "part_{}".format(part_id), fname))
                            vis_image.save(f"debug_vis/debug_image_{debug_count}.jpg")
                            if debug_count == 500:
                                assert False, "debug. "
                        if not args.debug:
                            vis_image.save(os.path.join(target_root, folder_name, "part_{}".format(part_id), fname))
                            # print("Saved.", os.path.join(target_root, folder_name, "part_{}".format(part_id), fname))

            if count % 10 == 0:
                print('{:.2f} \% done.'.format(count/len(code_list) * 100), flush=True)
    print("Done. ")
