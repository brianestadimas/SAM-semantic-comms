# Segment Anything based for Semantic Communications

This repository contains the implementation of a semantic segmentation model called Segment Anything based for Semantic Communications. It uses a VIT-H model pre-trained on COCO-Stuff dataset for segmenting images and videos.

## Steps

To run the model, please follow the below steps:

1. Download the VIT-H model weights from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
2. Create a conda environment using the provided `environment.yml` file:
`conda env create -f environment.yml`

3. Create a `weights` folder in the root directory of the project and put the downloaded VIT-H model weights file inside it.

## Steps for Predicting Particles in a Video

To predict particles in a video, please follow the below steps:

1. Run the following command in the terminal:
`python debugger.py`

2. Place the input video file `video.avi` into the `output/files` directory.
3. Run the following command in the terminal to start the transmitter:
`python video_transmitter.py`

## Utilities

### Creating an Input Video

To create an input video, run the following command in the terminal:
`python debugger.py`

### Exporting the Conda Environment

To export the conda environment used in this project, run the following command in the terminal:
`conda env export | grep -v "^prefix: " > environment.yml`
