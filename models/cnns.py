from torch import nn


class DeepQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 128, 4, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8 * 8 * 128, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        out_1 = nn.functional.elu(self.bn1(self.conv1(x)))
        out_2 = nn.functional.elu(self.bn2(self.conv2(out_1)))
        out_3 = nn.functional.elu(self.bn3(self.conv3(out_2)))
        out_4 = nn.functional.elu(self.fc1(out_3.view(x.shape[0], -1)))
        out_5 = self.fc2(out_4)
        return out_5