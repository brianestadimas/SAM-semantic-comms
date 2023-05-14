from sam_fedforward import SAMGenerator

if __name__ == "__main__":
    
    # Whole Image Mask Annotator
    sam_generator = SAMGenerator(image_name = "0.17 s.png")
    # sam_generator.get_box_prompter(x = 138, y = 190, width = 400, height = 400)
    
    results = sam_generator.video_predict(
        source="output/particles/video.avi",
        points_per_side=14, 
        points_per_batch=64,
        min_mask_region_area=1000,
        output_path="output.avi",
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=1,
        box_nms_thresh=0.7,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=0.3413333333333333,
        crop_n_points_downscale_factor=1,
    )