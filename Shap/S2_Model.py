import os
from multiprocessing.spawn import freeze_support

import numpy
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
import json
from detectron2.data import DatasetCatalog, MetadataCatalog

import cv2

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator


class S2_Model:
    def __init__(self):
        self.count = 0
        freeze_support()

        register_coco_instances("mol_val", {}, r"Shap/data_s2/eval1.json",
                                r"Shap/data_s2")

        mol_metadata = MetadataCatalog.get("mol_val")

        cfg = get_cfg()
        # cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))  # "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml
        cfg.DATASETS.TEST = ("mol_val",)
        cfg.DATASETS.TRAIN = ("mol_val",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MIN_SIZE_TEST = 300
        cfg.MODEL.PIXEL_MEAN = [127, 127, 127]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128, 256]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
        cfg.MODEL.RPN.NMS_THRESH = 0.7
        cfg.MODEL.WEIGHTS = r"Shap/data_s2/backup.pth"
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 200
        cfg.TEST.DETECTIONS_PER_IMAGE = 100
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 3
        cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 0

        self.predictor = DefaultPredictor(cfg)
        self.evaluator = COCOEvaluator("mol_val", cfg, False, output_dir="./output/")

    def model(self, imgs):
        # with open(r"C:\Users\Laptop\Desktop\DECT\eval.json") as json_file:
        #     data = json.load(json_file)

        # val_loader = build_detection_test_loader(cfg, "mol_val")
        # results = inference_on_dataset(predictor.model, val_loader, evaluator)
        inputs = [{}]
        inputs[0]["image_id"] = 1

        APs = numpy.zeros((len(imgs), 1))

        for i, img in enumerate(imgs):
            outputs = [self.predictor(img)]
            # del outputs[0]["instances"]._fields["pred_keypoints"]
            # del outputs[0]["instances"]._fields["pred_keypoint_heatmaps"]
            if len(outputs[0]["instances"]._fields["pred_classes"]) > 0:
                APs[i] = 1
            # self.evaluator.reset()
            # self.evaluator.process(inputs, outputs)
            # results = self.evaluator.evaluate()
            # APs[i] = results["bbox"]["AP"]

        APs = numpy.nan_to_num(APs, copy=True, nan=0)

        print(self.count)
        self.count += 1

        return APs
