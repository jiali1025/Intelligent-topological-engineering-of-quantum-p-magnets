import inspect
import os

import cv2
import numpy
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


class KeypointDetection():
    """ Input: img 2D array, global x, global y for the small scale img
        Output: A array of [boolean, x, y], where boolean indicates whether it is target module,
                x,y is global position of molecule in nanometer
    """

    def __init__(self, img, global_position, scan_size):

        img = numpy.array(img, dtype=numpy.float)
        img_0 = numpy.zeros_like(img)
        cv2.normalize(img, img_0, 0, 255, cv2.NORM_MINMAX)

        self.img_0 = img_0
        self.global_pos = global_position
        self.path = os.path.dirname(os.path.relpath(inspect.getfile(self.__class__)))
        self.scan_size = scan_size  # size of the scan region in nanometer (small scale img)

    def cfg_info(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MIN_SIZE_TEST = 300
        cfg.MODEL.PIXEL_MEAN = [127, 127, 127]
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128, 256]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[1.0]]
        cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[0]]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
        cfg.MODEL.RPN.NMS_THRESH = 0.5
        cfg.MODEL.WEIGHTS = os.path.join(self.path, "KeypointDetection.pth")
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 10000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 2000
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 3
        return cfg

    def pos_convert(self, keypoints, W, H):
        """Chosen one reaction site and Convert to global position"""

        if len(keypoints) == 0:
            pos = [0, 0, 0]
            return pos

        rand_num = numpy.random.randint(0, 2)
        keypoint = keypoints[0][rand_num].tolist()

        local_x = keypoint[0]
        local_y = keypoint[1]

        local_pos = [local_x / W - 0.5, - local_y / H + 0.5]  # relative position counted from image center
        global_x = local_pos[0] * self.scan_size[0] + self.global_pos[0]
        global_y = local_pos[1] * self.scan_size[1] + self.global_pos[1]

        pos = [1, global_x, global_y]
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
        keypoints = instances["pred_keypoints"]  # [N,3] list of [x,y,c]

        pos = self.pos_convert(keypoints, W, H)  # converting the data structure of output

        return pos
