# Copyright (c) Facebook, Inc. and its affiliates.
# From https://github.com/facebookresearch/Detic/blob/main/detic/predictor.py. 
# Modified by Jang Hyun Cho.

import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import BitMasks, Instances
from Detic.detic.modeling.utils import reset_cls_test
from continuously_postprocess_dcrf import dense_crf

class CustomPredictor(DefaultPredictor):
    """
    D2's DefaultPredictor but takes arbitrary input as argument and add to input to model. 
    """
    def reshape_image(self, image):
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            image = image[:, :, ::-1]
        image = self.aug.get_transform(image).apply_image(image)
        image = image[:, :, ::-1]
        return image 

    def __call__(self, original_image, additional_input={}):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            height, width = image.shape[1:]

            inputs = {"image": image, "height": height, "width": width}
            inputs.update(additional_input)
            predictions = self.model([inputs])[0]
            return predictions



def get_clip_embeddings(vocabulary, prompt='a '):
    from Detic.detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

class PartVisualizationDemo(object):
    def __init__(self, object_cfg, part_cfg, args, 
        instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            object_cfg (CfgNode):
            part_cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        if args.vocabulary == 'custom':
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(',')
            classifier = get_clip_embeddings(self.metadata.thing_classes)
        else:
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.dcrf = args.dcrf 
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.object_predictor = AsyncPredictor(object_cfg, num_gpus=num_gpu)
            self.part_predictor = AsyncPredictor(part_cfg, num_gpus=num_gpu)
        else:
            self.object_predictor = CustomPredictor(object_cfg)
            self.part_predictor = CustomPredictor(part_cfg)
        reset_cls_test(self.object_predictor.model, classifier, num_classes)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        object_prediction = self.object_predictor(image)

        masks = object_prediction["instances"].pred_masks
        scores = object_prediction["instances"].scores
        topk_idxs = scores.topk(1)[1].flatten()
        masks_selected = masks[topk_idxs]
        part_instance = Instances(object_prediction["instances"].image_size)
        part_instance.gt_masks = BitMasks(masks_selected)
        part_instance.gt_classes = torch.zeros(1)
        object_instance = Instances(object_prediction["instances"].image_size)
        object_instance.gt_masks = BitMasks(masks_selected)
        object_instance.gt_classes = torch.zeros(1)
        object_input = {"instances": object_instance, "part_instances": part_instance}
        # print(masks_selected.shape, object_prediction["instances"].image_size, image.shape, flush=True)
        predictions = self.part_predictor(image, object_input)


        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = self.part_predictor.reshape_image(image)
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        instances = predictions["proposals"].to(self.cpu_device)
        if self.dcrf:
            bmask = instances.pred_masks
            num_c = bmask.shape[0] 
            cmask = (bmask * (torch.arange(num_c) + 1)[:, None, None]).sum(0)
            cmask = torch.tensor(dense_crf(image, cmask, num_c + 1))
            o_cls = cmask.unique() 
            o_cls = o_cls[o_cls != 0]
            bmask = torch.zeros(len(o_cls), *cmask.shape).bool() 
            for i, c in enumerate(o_cls):
                bmask[i] = cmask == c
            instances.pred_masks = bmask
        vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = CustomPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5