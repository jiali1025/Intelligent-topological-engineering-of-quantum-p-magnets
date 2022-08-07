import numpy
import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(24 * 16 * 16, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 3)

    def forward(self, x):
        # x [N, 1, 128, 128]
        x = self.pool(nn.functional.relu(self.conv1(x)))  # [N, 6, 64, 64]
        x = self.pool(nn.functional.relu(self.conv2(x)))  # [N, 12, 32, 32]
        x = self.pool(nn.functional.relu(self.conv3(x)))  # [N, 24, 16, 16]
        x = torch.flatten(x, start_dim=1)  # flatten all dimensions except batch [N , 24*16*16]
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = numpy.load(r'Running_example/DL_modules/Data/Rxn/dataset_train.npy')
dataset = torch.from_numpy(dataset)


trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)


classes = ('reaction', 'missing', 'no_reaction')
net = Net()
net.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12000, gamma=0.1)
net.eval()

for epoch in range(25):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        net.train()
        # get the inputs; data is a list of [inputs, labels]
        # labels = torch.zeros(5,2).to(device=device, dtype=torch.long)
        # for m in range(5):
        #     labels[m, data[m,0].to(dtype=torch.long)]= 1
        labels = data[:, 0].to(device=device, dtype=torch.long)
        inputs = data[:, 1:].view(-1, 1, 128, 128).to(device=device, dtype=torch.float)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item()

        net.eval()
        if i % 100 == 99:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.4f' %
                  (epoch + 1, i + 1, running_loss / 100))

            running_loss = 0.0


print('Finished Training')

PATH = 'Running_example/DL_modules/Data/Rxn/mymodel.pth'
torch.save(net.state_dict(), PATH)
