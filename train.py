from semantic_comms import calculate_psnr, train_semantic_communication_system, load_images, add_awgn_noise
import torch
import numpy as np
from semantic_models.sc1 import SemanticEncoder1, SemanticDecoder1
from semantic_models.sc2 import SemanticEncoder2, SemanticDecoder2
from semantic_models.tc import SemanticEncoderTC, SemanticDecoderTC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Images
    images = load_images("output/tc_ori")
    num_images = images.size(0)
    image_channels = images.size(1)
    image_height = images.size(2)
    image_width = images.size(3)

    # Reshape images tensor to fit the semantic encoder input shape
    images = images.view(num_images, image_channels, image_height, image_width)

    # Set the desired SNR and number of training epochs
    snr = 4  # dB
    num_epochs = 100

    # Train the semantic communication system
    encoder_model = SemanticEncoder1()
    decoder_model = SemanticDecoder1()
    
    # encoder_model = SemanticEncoder2()
    # decoder_model = SemanticDecoder2()
    
    # encoder_model = SemanticEncoderTC()
    # decoder_model = SemanticDecoderTC()

    encoder, decoder = train_semantic_communication_system(encoder_model, decoder_model, images, snr, num_epochs)

    # Calculate SNR
    encoder_images = encoder(images)
    noisy_images = add_awgn_noise(encoder_images, snr)
    restored_images = decoder(noisy_images)
    theta = 1.0
    
    snr = calculate_psnr(images, restored_images, restored_images, theta)


    print(images.shape)
    print(encoder_images.shape)
    print(restored_images.shape)
    print(f"PSNR over AWGN channel: {snr.item()} dB")
    