import cv2
import os

HOME = os.getcwd()
def convert_images_to_video(folder_path, image_extension='.png', output='video.avi', sort=True):
    image_folder = f"data/{folder_path}"
    video_name = output
    images = [img for img in os.listdir(image_folder) if img.endswith(image_extension)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    

def load_video(video_path, output_path="output.avi"):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    return cap, out