import cv2
import time
import argparse
from ultralytics import YOLO
from collections import defaultdict
import matplotlib.pyplot as plt
from gooey import Gooey, GooeyParser
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import torch
import numpy as np
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Import the model and feature extractor
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas",low_cpu_mem_usage=True).to(device)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

def resize_frame(frame, scale_percent=75):
    # Calculate the new dimensions
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    # Resize the frame
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return resized_frame


def image2formatted(image):
    inputs = feature_extractor(images=resize_frame(image), return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Get the dimensions of the original image
    original_height, original_width, _ = image.shape

    # Interpolate the predicted depth map to match the original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(original_height, original_width),
        mode="bicubic",
        align_corners=False,
    )

    # Visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    return formatted

@Gooey
def main():
    parser = GooeyParser(description="Object Detection using YOLOv8")

    parser.add_argument("video_path", help="Path to input video file", widget="FileChooser")
    parser.add_argument("output_path", help="Path to save output video file", widget="DirChooser")
    parser.add_argument("output_name", help="Name of the output video file")

    args = parser.parse_args()

    # Load the YOLOv8 model


    # Open the video file
    cap = cv2.VideoCapture(args.video_path)

    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()

    # Define the codec and create VideoWriter object
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path = args.output_path + "/" + args.output_name + ".mp4"
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Measure frame processing time
            start_frame_time = time.time()

            # Run YOLOv8 inference on the frame
            annotated_frame = np.array(image2formatted(frame))
            depth_map = annotated_frame
            normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
            heatmap = plt.cm.inferno(normalized_depth_map)
            heatmap_bgr = cv2.cvtColor((heatmap[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


            # Calculate frame processing time
            end_frame_time = time.time()
            frame_processing_time = end_frame_time - start_frame_time

            # Increment frame count for FPS calculation
            frame_count += 1

            # Display FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(heatmap_bgr, f"FPS: {round(fps)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Write the annotated frame to the output video
            out.write(heatmap_bgr)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", heatmap_bgr)
            cv2.putText(depth_map, f"FPS: {round(fps)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
