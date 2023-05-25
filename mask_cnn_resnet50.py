import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
import numpy as np
import os
from PIL import Image
import cv2

def split_image_into_masks(image_path, output_dir):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)

    # Load pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    with torch.no_grad():
        # Forward pass through the model
        predictions = model([image_tensor])

    masks = predictions[0]['masks'].detach().cpu()
    num_masks = masks.shape[0]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a binary mask for the background (non-masked area)
    background_mask = np.ones((image.height, image.width), dtype=np.uint8)
    for i in range(num_masks):
        mask = masks[i, 0].numpy()
        background_mask[mask > 0.5] = 0

    # Save the background mask as a separate image
    background_mask_path = os.path.join(output_dir, "background_mask.png")
    cv2.imwrite(background_mask_path, background_mask * 255)

    # Apply the background mask to the original image
    background = np.array(image) * np.expand_dims(background_mask, axis=2)

    # Save the background image
    background_image_path = os.path.join(output_dir, "background_image.png")
    cv2.imwrite(background_image_path, background)

    for i in range(num_masks):
        mask = masks[i, 0].numpy()

        # Copy colored pixels from the original image to the mask image
        colored_mask = np.array(image.copy())
        colored_mask[mask <= 0.5] = 0

        # Save the colored mask image to the output directory
        mask_filename = f"mask_{i}.png"
        mask_filepath = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_filepath, colored_mask)

    print(f"Split {num_masks} masks and saved to {output_dir}")


# Example usage
image_path = "data/airplane.jpg"
output_dir = "output/masks_rcnn"

split_image_into_masks(image_path, output_dir)