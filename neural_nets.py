import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self, n_inputs):
        super(ConvNet, self).__init__()
        self.image_size = 150
        self.nf = 2
        self.num_channels = 1
        self.n_inputs = n_inputs
        # Extract features maps
        self.Net1 = nn.Sequential(
            # input is (nc) x 60 x 60
            nn.Conv2d(self.num_channels, self.nf * 2, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # state size. (nf * 2) x 3 * 32
            nn.Conv2d(self.nf * 2, self.nf, 4, 2, 1, bias=False),
            nn.ReLU(),
        )
        # Reduce Dimension
        self.Net2 = nn.Sequential(
            # input size is (1024,)
            nn.Linear(648, 200, bias=False),
            nn.ReLU(),
        )

        # Prediction Q value based on concatenation of previous vector and phase
        self.Net3 = nn.Sequential(
            nn.Linear(200 + self.n_inputs, 30, bias=False),
            nn.ReLU(),
            nn.Linear(30, 1, bias=False),
        )

    def forward(self, state, img):

        if len(img.size()) == 3:
            batch_size = img.size(0)
        else:
            batch_size = 1
            state = state.reshape(1, state.size()[0])
        img = img.reshape(batch_size, 1, self.image_size, self.image_size)
        img = torch.tensor(img, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        feat_img = self.Net1(img)
        feat_img = feat_img.view(feat_img.size()[0], -1).squeeze(1)
        feat_img = self.Net2(feat_img)
        x = torch.cat([state, feat_img], 1)
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
