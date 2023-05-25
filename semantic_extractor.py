from sam_fedforward import SAMGenerator
import os
import cv2
import numpy as np

DEFAULT_PATH = "output/masks"

class SemanticBased():
    def __init__(self, image_name):
        self.sam_gen = SAMGenerator(image_name)
        self.image_name = image_name
    
    def extract_images(self, color_image=False):
        sam_result, image_bgr = self.sam_gen.get_masks_annotator(visualize=False)
        
        background_mask = np.ones_like(image_bgr, dtype=bool)
        for detection in sam_result:
            mask = detection["segmentation"]
            background_mask[mask] = False
        background_image = np.zeros_like(image_bgr)
        background_image[background_mask] = image_bgr[background_mask]
        if color_image:
            cv2.imwrite(f"{DEFAULT_PATH}/background.png", background_image)
        else:
            background_image_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{DEFAULT_PATH}/background.png", background_image_gray)
    
        # Loop through the sam_result dictionary and save each mask as a separate image
        for i, detection in enumerate(sam_result):
            mask = detection["segmentation"]
            mask_image = np.zeros_like(image_bgr)  # Create a black image of the same size as the original image
            mask_image[mask] = image_bgr[mask]  # Copy the colored pixels from the original image to the mask image
            
            if color_image:
                cv2.imwrite(f"{DEFAULT_PATH}/mask_{i}.png", mask_image)
            else:
                mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)  # Convert the mask image to grayscale
                cv2.imwrite(f"{DEFAULT_PATH}/mask_{i}.png", mask_image_gray)   
 
    def semantic_extractor(self):
        return