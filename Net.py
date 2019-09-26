import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, imsize):
        super(Net, self).__init__()
        self.downsample = ((imsize-4)/2)
        if self.downsample % 2 == 0:
            self.finalsize = int((self.downsample-4)/2)
        else:
            self.finalsize = int((self.downsample - 5) / 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * self.finalsize * self.finalsize, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.finalsize * self.finalsize)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = Net()