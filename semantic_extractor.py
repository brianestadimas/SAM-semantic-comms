from sam_fedforward import SAMGenerator
import os
import cv2

DEFAULT_PATH = "output/masks"

class SemanticBased():
    def __init__(self, image_name):
        self.sam_gen = SAMGenerator(image_name)
        self.image_name = image_name
    
    def extract_images(self, color_image=False):
        sam_result, image_bgr = self.sam_gen.get_masks_annotator(visualize=False)
        
        # Convert the image to grayscale
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Loop through the sam_result dictionary and save each mask as a separate image
        for i, detection in enumerate(sam_result):
            mask = detection["segmentation"]
            mask_image = (mask * 255).astype("uint8")
            
            # Apply the mask to the grayscale image
            result_image = cv2.bitwise_and(image_gray, image_gray, mask=mask_image)
            
            # Convert the result image to color if color_image is True
            if color_image:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
            
            # Save the result image
            cv2.imwrite(f"{DEFAULT_PATH}/mask_{i}.png", result_image)
    
    def semantic_extractor(self):
        return