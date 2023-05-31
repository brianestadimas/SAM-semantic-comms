from sam_fedforward import SAMGenerator
import os
import numpy as np
import cv2

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


folder_path = "output/original"
image_name = os.listdir(folder_path)[0]

sam_generator = SAMGenerator(image_name = image_name, image_path=os.path.join(folder_path, image_name))

sam_generator.get_middle_point_prompter()



# import os
# import numpy as np
# import cv2
# from sam_fedforward import SAMGenerator

# def extract_images(color_image=False):
#     # Load CIFAR-10 dataset
#     cifar_images = np.load('cifar-10-images.npy')  # Assuming you have the CIFAR-10 dataset saved as 'cifar-10-images.npy'

#     # Create output directories
#     os.makedirs("cifar-10-batches-foreground", exist_ok=True)
#     os.makedirs("cifar-10-batches-background", exist_ok=True)

#     for i, image_bgr in enumerate(cifar_images):
#         sam_generator = SAMGenerator(image_bgr=image_bgr)
#         sam_result, _ = sam_generator.get_masks_annotator(visualize=False)

#         background_mask = np.ones_like(image_bgr, dtype=bool)
#         for detection in sam_result:
#             mask = detection["segmentation"]
#             background_mask[mask] = False
#         background_image = np.zeros_like(image_bgr)
#         background_image[background_mask] = image_bgr[background_mask]

#         if color_image:
#             cv2.imwrite(f"cifar-10-batches-background/background_{i}.png", background_image)
#         else:
#             background_image_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
#             cv2.imwrite(f"cifar-10-batches-background/background_{i}.png", background_image_gray)

#         for j, detection in enumerate(sam_result):
#             mask = detection["segmentation"]
#             mask_image = np.zeros_like(image_bgr)
#             mask_image[mask] = image_bgr[mask]

#             if color_image:
#                 cv2.imwrite(f"cifar-10-batches-foreground/foreground_{i}_{j}.png", mask_image)
#             else:
#                 mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
#                 cv2.imwrite(f"cifar-10-batches-foreground/foreground_{i}_{j}.png", mask_image_gray)

# extract_images(color_image=True)