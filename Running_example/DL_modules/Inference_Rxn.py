from multiprocessing.spawn import freeze_support

import cv2
import numpy
import torch

from torch import nn

img = cv2.imread(r"Shap/data_s3/1.png")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img = numpy.zeros_like(img)
for i in range(3):
    img[:, :, i] = gray

img = cv2.resize(img[:, :, 0], (128, 128))
img = torch.from_numpy(img).view(1, 1, 128, 128).float()


class CNN_struc(nn.Module):
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


cnn_model = CNN_struc()
cnn_model.load_state_dict(torch.load(r"Shap/data_s3/ReactionClassification.pth"))

outputs = cnn_model(img)
f = torch.nn.Softmax()
outputs = f(outputs)

print(outputs)
