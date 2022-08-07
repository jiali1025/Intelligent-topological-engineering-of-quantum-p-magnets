from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import logging
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import HookBase
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy

# ========================================================================================== #

# register dataset used for training
register_coco_instances("mol_train", {}, r"Running_example/DL_modules/Data/Dect/dataset_train.json",
                        r"Running_example/DL_modules/Data/Dect/img_train")

cfg = get_cfg()  # get default trainer configurations
cfg.MODEL.DEVICE = "cpu"  # set training device
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))  # use a pretrained model

cfg.DATASETS.TRAIN = ("mol_train",)
cfg.DATASETS.TEST = ("mol_train",)
cfg.DATALOADER.NUM_WORKERS = 0

cfg.SOLVER.IMS_PER_BATCH = 4  # set batch size
cfg.SOLVER.BASE_LR = 0.005  # set base learning rate
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.MOMENTUM = 0.9
cfg.SOLVER.WEIGHT_DECAY = 0.001
cfg.SOLVER.MAX_ITER = 1000  # model can usually converge within 1000 iterations
cfg.SOLVER.STEPS = (800,)  # reduce LR at 800 itr

cfg.INPUT.CROP.ENABLED = False  # crop is already implemented in augmentation
cfg.INPUT.MAX_SIZE_TRAIN = 600  # regulated image size for training and testing
cfg.INPUT.MAX_SIZE_TEST = 600
cfg.INPUT.MIN_SIZE_TRAIN = (200, 500)
cfg.INPUT.MIN_SIZE_TEST = 400
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'

cfg.MODEL.PIXEL_MEAN = [127, 127, 127]  # we provide b&w images in dataset

# set model hyper-parameters
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # targets mol and other similar mol
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.4, 0.9]
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]  # the shape of mol can be bounded in a square box
cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 10000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
cfg.MODEL.RPN.NMS_THRESH = 0.5
cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = numpy.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks


trainer = Trainer(cfg)

trainer.train()  # start training
