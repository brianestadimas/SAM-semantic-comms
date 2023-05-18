import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Define semantic encoder and decoder models
class SemanticEncoder(nn.Module):
    def __init__(self):
        super(SemanticEncoder, self).__init__()

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

class SemanticDecoder(nn.Module):
    def __init__(self):
        super(SemanticDecoder, self).__init__()

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


# Define AWGN channel
def add_awgn_noise(signal, snr):
    signal_power = torch.mean(torch.square(signal))
    noise_power = signal_power / snr

    noise = torch.randn(signal.size()) * torch.sqrt(noise_power)
    noisy_signal = signal + noise

    return noisy_signal


# Define training loop
def train_semantic_communication_system(images, snr, num_epochs):
    encoder = SemanticEncoder()
    decoder = SemanticDecoder()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Semantic encoding
        encoded_images = encoder(images)

        # Compression (optional)
        compressed_images = torch.flatten(encoded_images, start_dim=1)

        # Transmission over AWGN channel
        noisy_images = add_awgn_noise(compressed_images, snr)

        # Data restoration
        restored_images = decoder(noisy_images)

        # Semantic decoding

        # Calculate loss
        loss = criterion(restored_images, images)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    return encoder, decoder


# Calculate SNR
def calculate_snr(original_signal, noisy_signal):
    signal_power = torch.mean(torch.square(original_signal))
    noise_power = torch.mean(torch.square(original_signal - noisy_signal))
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

# Example usage

# Images
num_images = images.size(0)
image_channels = images.size(1)
image_height = images.size(2)
image_width = images.size(3)

# Reshape images tensor to fit the semantic encoder input shape
images = images.view(num_images, image_channels, image_height, image_width)

# Set the desired SNR and number of training epochs
snr = 20  # dB
num_epochs = 100

# Train the semantic communication system
encoder, decoder = train_semantic_communication_system(images, snr, num_epochs)

# Calculate SNR
restored_images = decoder(encoder(images))
snr = calculate_snr(images, restored_images)

print(f"SNR over AWGN channel: {snr.item()} dB")