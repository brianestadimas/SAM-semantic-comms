import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


# Define AWGN channel
def add_awgn_noise(signal, snr):
    signal_power = torch.mean(torch.square(signal))
    noise_power = signal_power / snr

    noise = torch.randn(signal.size()) * torch.sqrt(noise_power)
    noisy_signal = signal + noise

    return noisy_signal


# Define training loop
def train_semantic_communication_system(encoder_model, decoder_model, images, snr, num_epochs):
    encoder = encoder_model
    decoder = decoder_model
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Semantic encoding
        encoded_images = encoder(images)

        # Compression (optional)
        # compressed_images = torch.flatten(encoded_images, start_dim=1)
        # compression_rate = 0.9
        # compressed_images = encoded_images.clone()
        # image_height, image_width = compressed_images.size(2), compressed_images.size(3)
        # roi_height = int(image_height * compression_rate)
        # roi_width = int(image_width * compression_rate)
        # compressed_images[:, :, :roi_height, :roi_width] = 0

        # Transmission over AWGN channel
        #noisy_images = add_awgn_noise(encoded_images, snr)

        # Data restoration
        restored_images = decoder(encoded_images)

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


# Calculate PSNR
def calculate_psnr(original_signal, roi_signal, roni_signal, theta=1.0):
    mse_roi = torch.mean(torch.square(original_signal - roi_signal))
    mse_roni = torch.mean(torch.square(original_signal - roni_signal))
    
    signal_power = torch.square(torch.max(original_signal))
    noise_power = mse_roi * theta # + mse_roni * (1 - theta)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

# Load Images
def load_images(folder_path):
    # target_size = (512, 512)
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path).convert("RGB")
            # image = image.resize(target_size)  # Resize image to target size
            image = transforms.ToTensor()(image)
            image_list.append(image)
    images = torch.stack(image_list)
    return images
