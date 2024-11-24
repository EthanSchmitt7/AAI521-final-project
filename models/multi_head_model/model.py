import torch
import torch.nn as nn


# Define a torch model
class MultiHeadModel(nn.Module):
    def __init__(self, group_sizes: list[int] = None):
        super(MultiHeadModel, self).__init__()
        # Number of class groups
        self.group_sizes = [3, 2, 2, 2, 4, 2, 3, 7, 3, 3, 6] if group_sizes is None else group_sizes
        self.n = len(self.group_sizes)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layers
        self.flatten = nn.Flatten()

        self.fc1 = nn.ModuleList([])
        self.fc2 = nn.ModuleList([])
        self.fc3 = nn.ModuleList([])
        for n in range(self.n):
            self.fc1.append(nn.Linear(in_features=128 * 8 * 8, out_features=128))
            self.fc2.append(nn.Linear(in_features=128, out_features=64))
            self.fc3.append(nn.Linear(in_features=64, out_features=self.group_sizes[n]))

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def cnn_block(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.dropout(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.dropout(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.dropout(x)

        # Fully connected layers
        x = self.flatten(x)

        return x

    def class_head(self, x, n):
        x = self.relu(self.fc1[n](x))
        x = self.relu(self.fc2[n](x))
        x = self.softmax(self.fc3[n](x))

        return x

    def forward(self, x):
        # Convolutional Block
        conv_output = self.cnn_block(x)

        x_groups = []
        for n in range(self.n):
            x_groups.append(self.class_head(conv_output, n))

        c1 = x_groups[0]
        c2 = c1[:, 1].unsqueeze(dim=1) * x_groups[1]
        c3 = c2[:, 1].unsqueeze(dim=1) * x_groups[2]
        c4 = torch.sum(c3, axis=1).unsqueeze(dim=1) * x_groups[3]

        c10 = c4[:, 0].unsqueeze(dim=1) * x_groups[9]
        c11 = torch.sum(c10, axis=1).unsqueeze(dim=1) * x_groups[10]
        c5 = (c4[:, 1].unsqueeze(dim=1) + torch.sum(c11, axis=1).unsqueeze(dim=1)) * x_groups[4]

        c9 = c2[:, 0].unsqueeze(dim=1) * x_groups[8]
        c6 = torch.sum(c9, axis=1).unsqueeze(dim=1) * x_groups[5]

        c7 = c1[:, 0].unsqueeze(dim=1) * x_groups[6]

        c8 = c6[:, 0].unsqueeze(dim=1) * x_groups[7]

        return torch.concat((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11), dim=1)
