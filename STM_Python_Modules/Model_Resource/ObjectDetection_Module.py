import inspect
import os
import cv2

import numpy
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import RandomFieldFilter


class ObjectDetection():
    """ Input: img 2D array, global x, global y for the large scale img
        Output: A 2D List of [N,4], where N is num of molecules found, 4 is global position of molecule in nanometer
                in [x, y, W, H], * W,H is currently setting to 0 as not required
    """

    def __init__(self, img, global_position, scan_size):

        img = numpy.array(img, dtype=numpy.double)
        img_0 = numpy.zeros_like(img)
        cv2.normalize(img, img_0, 0, 255, cv2.NORM_MINMAX)

        self.img_0 = img_0
        self.global_pos = global_position
        self.path = os.path.dirname(os.path.relpath(inspect.getfile(self.__class__)))
        self.scan_size = scan_size  # size of the scan region in nanometer (large scale img)

    def cfg_info(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.INPUT.MIN_SIZE_TEST = 400
        cfg.MODEL.PIXEL_MEAN = [127, 127, 127]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # set threshold for this model
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
        cfg.MODEL.RPN.NMS_THRESH = 0.5
        cfg.MODEL.WEIGHTS = os.path.join(self.path, "ObjectDetection.pth")
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 200
        cfg.TEST.DETECTIONS_PER_IMAGE = 100
        return cfg

    def pos_convert(self, bbxes, W, H):
        """Remove molecules on edge and convert to global position"""

        pos = []

        for bbox in bbxes:
            check_edge_1 = bbox == 0
            check_edge_2 = bbox == W
            check_edge_3 = bbox == H
            check_edge = sum(check_edge_1 + check_edge_2 + check_edge_3)

            if check_edge == 0:  # remove molecules on edge
                local_x = (bbox[0] + bbox[2]) / 2
                local_y = (bbox[1] + bbox[3]) / 2
                local_center = [local_x / W - 0.5, - local_y / H + 0.5]  # relative position counted from image center
                global_x = local_center[0] * self.scan_size[0] + self.global_pos[0]
                global_y = local_center[1] * self.scan_size[1] + self.global_pos[1]
                pos.extend([global_x.tolist(), global_y.tolist(), 4.0, 4.0])

        return pos

    def prediction(self):
        cfg = self.cfg_info()
        predictor = DefaultPredictor(cfg)

        W, H = self.img_0.shape

        img_1 = numpy.zeros((W, H, 3))
        for i in range(3):
            img_1[:, :, i] = self.img_0  # Convert to 3 channels

        outputs = predictor(img_1)

        instances = outputs["instances"]._fields

        CRF = RandomFieldFilter.RF_filter(instances)
        instances = CRF.instances  # remove odd size

        num = len(instances["pred_classes"])

        bbxes = []

        for i in range(num):
            if instances["pred_classes"][i] == 1:
                bbxes.append(instances["pred_boxes"][i].tensor[0])  # x1 y1 x2 y2

        pos = self.pos_convert(bbxes, W, H)  # converting the data structure of output

        return pos
