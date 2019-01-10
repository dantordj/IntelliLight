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

    def __init__(self, n_inputs):
        super(LinearNet, self).__init__()
        self.n_inputs = n_inputs

        self.fc1 = nn.Linear(self.n_inputs, 1, bias=False)

    def forward(self, x):
        return self.fc1(x)


class DeepNet(nn.Module):

    def __init__(self, n_inputs):
        super(DeepNet, self).__init__()
        self.n_inputs = n_inputs

        self.fc1 = nn.Linear(self.n_inputs, 16, bias=False)
        self.activation = torch.nn.ReLU()
        self.fc2 = nn.Linear(16, 5, bias=False)
        self.fc3 = nn.Linear(5, 1, bias=False)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)
