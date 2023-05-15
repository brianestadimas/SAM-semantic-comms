from sam_fedforward import SAMGenerator
import os
import cv2

DEFAULT_PATH = "output/masks"

class SemanticExtractor():
    def __init__(self, image_name):
        self.sam_gen = SAMGenerator(image_name)
        self.image_name = image_name
    
    def output_semantic(self, color_image=False):
        sam_result, image_bgr = self.sam_gen.get_masks_annotator(visualize=False)
        # Loop through the sam_result dictionary and save each mask as a separate image
        for i, detection in enumerate(sam_result):
            mask = detection["segmentation"]
            mask_image = (mask * 255).astype("int16")
            color = (0, 255, 0) # Define the color here
            # if color_image:
                # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)  # Convert to color image
                # mask_color = cv2.merge([color if channel_mask.any() else channel for channel, channel_mask in zip(cv2.split(mask_image), cv2.split(mask))])  # Color the mask
                # mask_image = cv2.addWeighted(image_bgr, 0.5, mask_color, 0.5, 0)  # Blend with original image
            cv2.imwrite(f"{DEFAULT_PATH}/mask_{i}.png", mask_image)