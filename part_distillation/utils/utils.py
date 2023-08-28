# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pycocotools import mask as mask_util
from detectron2.utils.visualizer import ColorMode, Visualizer, GenericMask, _create_text_labels
import numpy as np
import torch
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as dcrf_utils



def proposals_to_coco_json(binary_mask):
    """
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(binary_mask)
    if num_instance == 0:
        return []

    rles = [mask_util.encode(np.array(part_mask[:, :, None], order="F", dtype="uint8"))[0]
            for part_mask in binary_mask]
    for rle in rles:
        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
        # json writer which always produces strings cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what
        # the pycocotools/_mask.pyx does).
        rle["counts"] = rle["counts"].decode("utf-8")

    return [{"segmentation": rle} for rle in rles]


def get_iou_all_cocoapi(pr_masks, gt_masks):
    pr_masks = pr_masks.cpu()
    gt_masks = gt_masks.cpu()
    pr_masks = [mask_util.encode(np.asfortranarray(m.cpu().numpy())) for m in pr_masks]
    gt_masks = [mask_util.encode(np.asfortranarray(m.cpu().numpy())) for m in gt_masks]
    ious = mask_util.iou(pr_masks, gt_masks, [0 for _ in range(len(gt_masks))])

    return torch.tensor(ious)

# def get_iou_all(msk, msk_all):
#     return (msk * msk_all).flatten(1).sum(1) / ((msk + msk_all).bool().flatten(1).sum(1) + 1e-10)




def dense_crf(image, label, n_labels, p=0.7, t=5, sd1=3, sd2=5, sc=13, compat1=3, compat2=10):
    annotated_label = label.to(torch.int32).numpy()
    colors, labels = np.unique(annotated_label, return_inverse=True)

    c = image.shape[2]
    h = image.shape[0]
    w = image.shape[1]

    d = dcrf.DenseCRF2D(w, h, n_labels)
    U = dcrf_utils.unary_from_labels(labels, n_labels, gt_prob=p, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    feats = dcrf_utils.create_pairwise_gaussian(sdims=(sd1, sd1), shape=(h, w))
    d.addPairwiseEnergy(feats, compat=compat1, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    feats = dcrf_utils.create_pairwise_bilateral(sdims=(sd2, sd2), schan=(sc, sc, sc),
                                                 img=image,
                                                 chdim=2)
    d.addPairwiseEnergy(feats, compat=compat2,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(t)
    Q = np.array(Q).reshape((n_labels, h, w)).argmax(axis=0)

    return Q





class Partvisualizer(Visualizer):
    def draw_instance_predictions(self, predictions):
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("part_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION:
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.7
        else:
            colors = None
            alpha = 0.6

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.6

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output
