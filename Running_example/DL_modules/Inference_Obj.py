from multiprocessing.spawn import freeze_support

import cv2
import numpy
from matplotlib import pyplot as plt

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

from detectron2.data import MetadataCatalog


# load an image
from detectron2.utils.visualizer import Visualizer

img_0 = cv2.imread(r"Shap/data_s1/1.png")
gray = cv2.cvtColor(img_0, cv2.COLOR_RGB2GRAY)
img = numpy.zeros_like(img_0)
for i in range(3):
    img[:, :, i] = gray  # Convert to grayscale for inference


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

predictor = DefaultPredictor(cfg)

outputs = predictor(img)
v = Visualizer(img[:, :, ::-1])
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Plot the image
plt.figure(figsize=(14, 10))
plt.imshow(out.get_image()[:, :, ::-1])
plt.show()