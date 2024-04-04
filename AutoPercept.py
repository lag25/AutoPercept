import cv2
import time
import argparse
from ultralytics import YOLO
from collections import defaultdict
# Import Gooey
from gooey import Gooey, GooeyParser


classes = {0 : 'Car',    ## Class with thier corresponding label
  1: 'Pedestrian',
 2 : 'Van',
  3 : 'Cyclist',
  4 : 'Truck',
  5:'Misc',
 6: 'Tram',
  7: 'Person_sitting'}

def count(tensor):
    res = defaultdict(int)
    for i in tensor:
        res[classes[int(i)]]+=1
    return res


@Gooey
def main():
    parser = GooeyParser(description="Object Detection using YOLOv8")

    parser.add_argument("model_path", help="Path to YOLOv8 model file", widget="FileChooser")
    parser.add_argument("video_path", help="Path to input video file", widget="FileChooser")
    parser.add_argument("output_path", help="Path to save output video file", widget="DirChooser")
    parser.add_argument("output_name", help="Name of the output video file")

    args = parser.parse_args()

    # Load the YOLOv8 model
    model = YOLO(args.model_path)

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
            results = model(frame)
            for r in results:
                f = count(r.boxes.cls)




            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            markers = [len(frame[2]) - 180, 50]
            for cls in f:
                clr = [100, 100, 100]

                # Get pixel intensity values at markers position
                pixel_intensity = frame[markers[1], markers[0]]

                # Compute inverse of pixel intensity values and convert to integers
                inverse_intensity = [int(255 - intensity)//2 for intensity in pixel_intensity]
                inverse_intensity[0] = 110
                inverse_intensity[1] = 209
                # Set color for text
                text_color = tuple(inverse_intensity)

                # Draw text on annotated frame with inverted color
                cv2.putText(annotated_frame, f"{cls} : {f[cls]}", markers, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                markers[1] += 25

            # Calculate frame processing time
            end_frame_time = time.time()
            frame_processing_time = end_frame_time - start_frame_time

            # Increment frame count for FPS calculation
            frame_count += 1

            # Display FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(annotated_frame, f"FPS: {round(fps)}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Write the annotated frame to the output video
            out.write(annotated_frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

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
