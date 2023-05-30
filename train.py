from semantic_comms import calculate_psnr, train_semantic_communication_system, load_images, add_awgn_noise, calculate_compression_rate
import torch
import numpy as np
from semantic_models.sc1 import SemanticEncoder, SemanticDecoder
from semantic_models.sc2 import SemanticEncoder2, SemanticDecoder2
from semantic_models.tc import SemanticEncoderTC, SemanticDecoderTC
from semantic_models.vit import SemanticEncoderVIT, SemanticDecoderVIT
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Images
    images = load_images("output/masks")
    num_images = images.size(0)
    image_channels = images.size(1)
    image_height = images.size(2)
    image_width = images.size(3)

    # Reshape images tensor to fit the semantic encoder input shape
    images = images.view(num_images, image_channels, image_height, image_width)

    # Set the desired SNR and number of training epochs
    snr = 13  # dB
    num_epochs = 1500

    # Train the semantic communication system
    encoder_model = SemanticEncoder()
    decoder_model = SemanticDecoder()

    encoder_model_vit = SemanticEncoderVIT()
    decoder_model_vit = SemanticDecoderVIT()

    encoder, decoder = train_semantic_communication_system(encoder_model, decoder_model, images, snr, num_epochs)

    # Calculate SNR
    encoder_images = encoder(images)
    compressed_size = encoder_images.numel() * encoder_images.element_size()  # Size of compressed representation in bytes
    
    noisy_images = add_awgn_noise(encoder_images, snr)
    restored_images = decoder(encoder_images)
    theta = 1.0
    
    snr = calculate_psnr(images, restored_images, theta)
    
    # Calculate the original image size in bytes
    original_size = images.numel() * images.element_size()

    # Calculate the compression rate
    compression_rate = calculate_compression_rate(original_size, compressed_size)
    print(compression_rate)

    print(images.shape)
    print(encoder_images.shape)
    print(restored_images.shape)
    print(f"PSNR over AWGN channel: {snr.item()} dB")
    