import numpy
import torch
import torch.nn as nn
import cv2


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


class S3_Model:
    def __init__(self):
        self.cnn_model = CNN_struc()
        self.cnn_model.load_state_dict(torch.load(r"Shap/data_s3/ReactionClassification.pth"))
        self.count = 0

    def model(self, imgs):
        losses = numpy.zeros((len(imgs), 1))

        for i, img in enumerate(imgs):
            img = cv2.resize(img[:, :, 0], (128, 128))
            img = torch.from_numpy(img).view(1, 1, 128, 128).float()
            outputs = self.cnn_model(img)
            f = torch.nn.Softmax()

            outputs = f(outputs)
            label = torch.tensor([2], dtype=torch.long)
            cel = torch.nn.CrossEntropyLoss()

            losses[i] = - cel(outputs, label).detach().numpy()

        print(self.count)
        self.count += 1

        return losses
