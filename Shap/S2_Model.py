from multiprocessing.spawn import freeze_support

import numpy
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

from detectron2.evaluation import COCOEvaluator


class S2_Model:
    def __init__(self):
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
        cfg.MODEL.WEIGHTS = r"Shap/data_s2/KeypointDetection.pth"
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 200
        cfg.TEST.DETECTIONS_PER_IMAGE = 100
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 3
        cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 0

        self.predictor = DefaultPredictor(cfg)
        self.evaluator = COCOEvaluator("mol_val", cfg, False, output_dir="./output/")

    def model(self, imgs):
        inputs = [{}]
        inputs[0]["image_id"] = 1

        true_points = numpy.array([
            [
                387.0,
                398.0
            ],
            [
                239.5,
                436.0
            ],
            [
                278.0,
                308.5
            ]
        ])

        loss = numpy.zeros((len(imgs), 1))

        for i, img in enumerate(imgs):
            outputs = [self.predictor(img)]
            if len(outputs[0]["instances"]._fields["pred_boxes"]) == 0:
                loss[i] = -100
            else:
                pred_points = outputs[0]["instances"]._fields["pred_keypoints"].cpu().numpy().squeeze()
                pred_points = pred_points[:, :2]

                loss1 = min(numpy.linalg.norm(pred_points[0, :] - true_points, axis=1))
                loss2 = min(numpy.linalg.norm(pred_points[1, :] - true_points, axis=1))
                loss3 = min(numpy.linalg.norm(pred_points[2, :] - true_points, axis=1))

                loss[i] = -(loss1 + loss2 + loss3)

        return loss
