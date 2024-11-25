import torch
import torch.nn as nn


class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_pieces):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            num_pieces (int): Number of linear pieces (k in maxout).
        """
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces

        # Create the weights and biases for all linear pieces
        self.linear = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x):
        # Compute the linear transformation
        output = self.linear(x)

        # Reshape to (batch_size, out_features, num_pieces)
        batch_size = output.shape[0]
        output = output.view(batch_size, self.out_features, self.num_pieces)

        # Take the max over the pieces
        output, _ = torch.max(output, dim=2)
        return output


# Define a torch model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=6, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)

        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layers
        self.flatten = nn.Flatten()

        # Maxout
        self.mo1 = Maxout(in_features=1152, out_features=1152, num_pieces=2)
        self.mo2 = Maxout(in_features=1152, out_features=37, num_pieces=2)

        # Activation functions
        self.relu = nn.ReLU()

    def cnn_block(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))

        # Fully connected layers
        x = self.flatten(x)

        return x

    def forward(self, x):
        # Convolutional Block
        conv_output = self.cnn_block(x)

        # Maxout Block
        mo1_output = self.mo1(conv_output)
        mo1_output = self.dropout(mo1_output)
        mo2_output = self.mo2(mo1_output)
        mo2_output = self.dropout(mo2_output)

        return mo2_output
