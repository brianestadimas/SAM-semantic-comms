from sam_fedforward import SAMGenerator

if __name__ == "__main__":
    
    # Whole Image Mask Annotator
    sam_generator = SAMGenerator(image_name = "0.17 s.png")
    # sam_generator.get_box_prompter(x = 138, y = 190, width = 400, height = 400)
    
    results = sam_generator.video_predict(
        source="output/particles/video.avi",
        points_per_side=16, 
        points_per_batch=64,
        min_area=100,
        output_path="output.avi",
    )