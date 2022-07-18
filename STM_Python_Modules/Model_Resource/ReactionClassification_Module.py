import inspect
import logging
import os

import cv2
import torch
import torch.nn as nn

from KeypointDetection_Module import KeypointDetection


class ReactionClassification():
    """ Input: img 2D array, global x, global y for the small scale img
        Output: [boolean, x, y], where boolean indicate whether to retry (true means no reaction, need retry),
                x,y the global position for new reaction site if the reaction is not successful.
                when moving on (boolean = false), 0 will be assigned to x,y
    """

    def __init__(self, img, global_x, global_y, scan_size):
        self.img_0 = img
        self.global_pos = [global_x, global_y]  # global x, global y for the small scale img
        self.path = os.path.dirname(os.path.relpath(inspect.getfile(self.__class__)))
        self.scan_size = scan_size

    def pos_convert(self):
        return 0

    def CNN_model(self, img):
        """CNN model
            Input: 1 channel img
            Output: [1] one class (0: successful reaction; 1: molecule missing; 2: no reaction )"""

        img_r = cv2.resize(img, (128, 128))

        model = CNN_structure()
        model_path = os.path.join(self.path, "ReactionClassification.pth")
        model.load_state_dict(torch.load(model_path))

        img_r = torch.from_numpy(img_r).view(1, 1, 128, 128).float()

        prediction = model(img_r)
        _, state = torch.max(prediction, 1)

        return state

    def call_keypoint_detection(self, img):
        """ call keypoint detection if retry is required
            input: 2D array, global x,y of small scale image"""

        kp_model = KeypointDetection(img, self.global_pos[0], self.global_pos[1], self.scan_size)
        pos = kp_model.prediction()

        return pos

    def log_outcome(self, status, info):
        """ log outcome after giving pulse
            INFO: Status, global position x,y of pulse"""

        filepath = os.path.join(self.path, "outcome.log")
        logging.basicConfig(filename=filepath, filemode='a', level=logging.INFO)
        logging.info("status: {}, position: {},{}".format(status, info[1], info[2]))

    def prediction(self):
        x, y = 0, 0

        reaction_status = self.CNN_model(self.img_0/255.0)

        if reaction_status == 0:  # reaction happens, move on
            retry = 0
        elif reaction_status == 1:  # missing molecule, move on
            retry = 0
        elif reaction_status == 2:  # no reaction, need try
            pos = self.call_keypoint_detection(self.img_0)
            retry = pos[0]
            if retry == 0:
                reaction_status = 1
            x = pos[1]
            y = pos[2]

        info_to_labelview = [retry, x, y]
        self.log_outcome(reaction_status, info_to_labelview)

        return info_to_labelview


class CNN_structure(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(24 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        # x [N, 1, 128, 128]
        x = self.pool(nn.functional.relu(self.conv1(x)))  # [N, 6, 64, 64]
        x = self.pool(nn.functional.relu(self.conv2(x)))  # [N, 12, 32, 32]
        x = self.pool(nn.functional.relu(self.conv3(x)))  # [N, 24, 16, 16]
        x = torch.flatten(x, start_dim=1)  # flatten all dimensions except batch [N , 24*16*16]
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
