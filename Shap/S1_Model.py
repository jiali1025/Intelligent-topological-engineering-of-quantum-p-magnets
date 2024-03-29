from multiprocessing.spawn import freeze_support

import numpy
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator


class S1_Model:
    def __init__(self):
        self.count = 0
        freeze_support()

        register_coco_instances("mol_val", {}, r"Shap/data_s1/eval1.json",
                                r"Shap/data_s1")

        mol_metadata = MetadataCatalog.get("mol_val")

        cfg = get_cfg()
        # cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.DATASETS.TEST = ("mol_val",)
        cfg.DATASETS.TRAIN = ("mol_val",)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.INPUT.MIN_SIZE_TEST = 400
        cfg.MODEL.PIXEL_MEAN = [127, 127, 127]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
        cfg.MODEL.RPN.NMS_THRESH = 0.5
        cfg.MODEL.WEIGHTS = r"Shap/data_s1/ObjectDetection.pth"
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 200
        cfg.TEST.DETECTIONS_PER_IMAGE = 100

        self.predictor = DefaultPredictor(cfg)
        self.evaluator = COCOEvaluator("mol_val", cfg, False, output_dir="./output/")

    def model(self, imgs):

        inputs = [{}]
        inputs[0]["image_id"] = 1

        APs = numpy.zeros((len(imgs), 1))

        for i, img in enumerate(imgs):
            outputs = [self.predictor(img)]
            self.evaluator.reset()
            self.evaluator.process(inputs, outputs)
            results = self.evaluator.evaluate()
            APs[i] = results["bbox"]["AP"]

        APs = numpy.nan_to_num(APs, copy=True, nan=0)

        print(self.count)
        self.count += 1

        return APs
