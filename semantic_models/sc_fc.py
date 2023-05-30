import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import functional as F
from vit_pytorch import ViT

# Define semantic encoder and decoder models
class SemanticEncoder(nn.Module):
    def __init__(self, compression_rate = 0.5):
        super(SemanticEncoder, self).__init__()

        # Define encoder architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(32 * 8 * 8, 256)  # Added fully connected layer
        self.compression_rate = compression_rate

    def forward(self, x):
        # Perform encoding
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        fc_input_size = x.size(1)  # Get the size of the flattened dimension
        self.fc1 = nn.Linear(fc_input_size, int(fc_input_size * self.compression_rate))  # Update the fully connected layer input size
        x = self.fc1(x)
        
        return x


class SemanticDecoder(nn.Module):
    def __init__(self):
        super(SemanticDecoder, self).__init__()

        # Define decoder architecture
        self.fc1 = nn.Linear(1024, 32 * 8 * 8)  # Added second fully connected layer


        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(3)

    def forward(self, x):
        # Perform decoding
        x = self.fc1(x)
        x = x.view(x.size(0), 32, 8, 8)

        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)

        return x