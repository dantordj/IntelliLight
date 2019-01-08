import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.image_size = 64
        self.nf = 8
        self.num_channels = 1

        # Extract features maps
        self.Net1 = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.nf, 4, 1, bias=False),
            nn.ReLU(),
            # state size. (nf) x 64 x 64
            nn.Conv2d(self.nf, self.nf * 2, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # state size. (nf / 2) x 32 * 32
            nn.Conv2d(self.nf * 2, self.nf / 2, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        # Reduce Dimension
        self.Net2 = nn.Sequential(
            # input size is (1024,)
            nn.Linear(1024, 200),
            nn.ReLU(),
        )

        # Prediction Q value based on concatenation of previous vector and phase
        self.Net3 = nn.Sequential(
            nn.Linear(201, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, img, phase):
        x_intermediate = self.Net1(img)
        x = x.view(-1).squeeze(1)
        x = self.Net2(x)
        x = torch.cat([x, [1]])
        x = self.Net3(x)

        return x


class LinearNet(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.feature_size = 6

        self.Net = nn.Sequential(
            nn.Linear(self.feature_size, 10, bias=False),
            torch.nn.ReLU(),
            nn.Linear(10, 1, bias=False),
        )

    def forward(self, x):
        x = self.Net(x)
        return x


class LinearNet2(nn.Module):

    def __init__(self):
        super(LinearNet, self).__init__()
        self.feature_size = 8

        self.Net = nn.Sequential(
            nn.Linear(self.feature_size, 2)
        )

    def forward(self, x):
        x = self.Net(x)
        return x
