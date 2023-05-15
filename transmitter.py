from semantic_extractor import SemanticExtractor

if __name__ == "__main__":
    # Semantic Extraction for masks and foreground
    semantic_images = SemanticExtractor(image_name="catdog.jpg")
    semantic_images.output_semantic(color_image=True)