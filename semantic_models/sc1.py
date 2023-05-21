import torch.nn as nn

# Define semantic encoder and decoder models
class SemanticEncoder1(nn.Module):
    def __init__(self):
        super(SemanticEncoder1, self).__init__()

        # Define encoder architecture
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

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

        return x

class SemanticDecoder1(nn.Module):
    def __init__(self):
        super(SemanticDecoder1, self).__init__()

        # Define decoder architecture
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu1 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu2 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(3)

    def forward(self, x):
        # Perform decoding
        x = self.deconv1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        x = self.deconv2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)

        return x