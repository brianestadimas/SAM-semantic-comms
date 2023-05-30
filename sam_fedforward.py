import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from tqdm import tqdm
from jupyter_bbox_widget import BBoxWidget
import os
import cv2
import supervision as sv
import numpy as np
import pandas as pd
from utils import load_video

HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Model Registry and Load Checkpoint
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

# helper function that loads an image before adding it to the widget
import base64
def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded


# Automatic Mask Generator
class SAMGenerator():
    def __init__(self, image_name, image_path):
        self.IMAGE_NAME = image_name
        self.IMAGE_PATH = os.path.join(HOME, "data", image_name) if not image_path else image_path
        '''
        SamAutomaticMaskGenerator returns a list of masks, where each mask is a dict containing various information about the mask:
        segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
        area - [int] - the area of the mask in pixels
        bbox - [List[int]] - the boundary box of the mask in xywh format
        predicted_iou - [float] - the model's own prediction for the quality of the mask
        point_coords - [List[List[float]]] - the sampled input point that generated this mask
        stability_score - [float] - an additional measure of mask quality
        crop_box - List[int] - the crop of the image used to generate this mask in xywh format
        '''

    def get_masks_annotator(self, visualize=True):
        # Load the image
        image_bgr = cv2.imread(self.IMAGE_PATH)
        
        # # save image_bgr matrix as txt  file
        # # np.savetxt("image_bgr.txt", image_bgr)
        # with open("image_bgr.txt", "w") as f:
        #     f.write(str(image_bgr))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Generate the masks
        mask_generator = SamAutomaticMaskGenerator(model=sam,
                                                   points_per_side=12, 
                                                   points_per_batch=64,
                                                   min_mask_region_area=200,
                                                   pred_iou_thresh=0.88,
                                                   stability_score_thresh=0.95,
                                                   stability_score_offset=1,
                                                   box_nms_thresh=0.7,
                                                   crop_nms_thresh=0.7,
                                                   crop_overlap_ratio=0.3413333333333333,
                                                   crop_n_points_downscale_factor=1)
        sam_result = mask_generator.generate(image_rgb)

        # print(sam_result[0].keys())
        # Visualize the mask with supervision
        if visualize:
            mask_annotator = sv.MaskAnnotator()
            detections = sv.Detections.from_sam(sam_result=sam_result)
            annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
            sv.plot_images_grid(
                images=[image_bgr, annotated_image],
                grid_size=(1, 2),
                titles=['source image', 'segmented image']
            )
        
        return sam_result, image_bgr
                
    def get_box_prompter(self, x = 68, y = 247, width = 555, height = 678, label = ''):
        '''
        SamAutomaticBoxGenerator returns a list of boxes, where each box is a dict containing various information about the box:
        x - [int] - the x coordinate of the top left corner of the box
        y - [int] - the y coordinate of the top left corner of the box
        width - [int] - the width of the box
        height - [int] - the height of the box
        label - [str] - the label of the box
        '''
        # Load the image
        image_bgr = cv2.imread(self.IMAGE_PATH)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Prompting Boxes
        widget = BBoxWidget()
        widget.image = encode_image(self.IMAGE_PATH)
        box = np.array([
            x, 
            y, 
            x + width, 
            y + height
        ])
        
        mask_predictor = SamPredictor(sam)
        mask_predictor.set_image(image_rgb)

        masks, scores, logits = mask_predictor.predict(
            box=box,
            multimask_output=True
        )
        box_annotator = sv.BoxAnnotator(color=sv.Color.red())
        mask_annotator = sv.MaskAnnotator(color=sv.Color.red())
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks
        )
        detections = detections[detections.area == np.max(detections.area)]

        source_image = box_annotator.annotate(scene=image_bgr.copy(), detections=detections, skip_label=True)
        segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

        sv.plot_images_grid(
            images=[source_image, segmented_image],
            grid_size=(1, 2),
            titles=['source image', 'segmented image']
        )

    def video_predict(self, source, points_per_side, points_per_batch, min_mask_region_area, stability_score_thresh, pred_iou_thresh, stability_score_offset,
                      box_nms_thresh, crop_nms_thresh, crop_overlap_ratio, crop_n_points_downscale_factor, output_path="output.avi"):
        centroids = pd.DataFrame(columns=['Batch', 'Mask', 'Centroid X', 'Centroid Y'])
        batch_num = 0
        
        cap, out = load_video(source, output_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            if not ret:
                break

            model = sam
            mask_generator = SamAutomaticMaskGenerator(
                model, points_per_side=points_per_side, points_per_batch=points_per_batch, 
                stability_score_thresh=stability_score_thresh, pred_iou_thresh=pred_iou_thresh, min_mask_region_area=min_mask_region_area,
                box_nms_thresh=box_nms_thresh, crop_nms_thresh=crop_nms_thresh, crop_overlap_ratio=crop_overlap_ratio, 
                crop_n_points_downscale_factor=crop_n_points_downscale_factor, stability_score_offset = stability_score_offset
            )
            masks = mask_generator.generate(frame)
        
            if len(masks) == 0:
                continue

            sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
            mask_image = np.zeros(
                (masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3), dtype=np.uint8
            )

            for i, ann in enumerate(sorted_anns):
                m = ann["segmentation"]
                color = colors[i % 256]
                img = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
                img[:, :, 0] = color[0]
                img[:, :, 1] = color[1]
                img[:, :, 2] = color[2]
                img = cv2.bitwise_and(img, img, mask=m.astype(np.uint8))
                img = cv2.addWeighted(img, 0.35, np.zeros_like(img), 0.65, 0)
                mask_image = cv2.add(mask_image, img)
                
                # calculate centroid of mask
                moments = cv2.moments(m.astype(np.uint8))
                if moments["m00"] == 0:
                    continue
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                # do something with centroid, such as print it out
                centroids.loc[len(centroids)] = [batch_num, i, cx, cy]
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx}, {cy})", (cx-30, cy-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            batch_num += 1
            combined_mask = cv2.add(frame, mask_image)
            out.write(combined_mask)

        out.release()
        cap.release()
        cv2.destroyAllWindows()
        centroids.to_excel('centroids_final.xlsx', index=False)
        
        return output_path

