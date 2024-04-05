# AutoPercept
A project focused on YOLO object detection trained on the KITTI dataset. This repository serves as a starting point for implementing object detection capabilities in autonomous systems using YOLO architecture.

## Overview
AutoPercept leverages the power of YOLO (You Only Look Once) for real-time object detection and tracking. The model is trained on the KITTI dataset, enabling accurate detection of objects in various driving scenarios.

## Key Features
- **YOLO Object Detection:** Real-time detection and tracking of objects using YOLO architecture.
- **Real Time Counting:** Functionality for counting and displaying the amount of detections for each class has been added
- **Pythonic UI** : AutoPercept can also be used as a Wrapper or Interface to run inference on videos using custom model weights.
- **Savable Results** : Functionality to save the video with the detections in a given directory with a given name has also been added
- **KITTI Dataset:** Trained on the KITTI dataset, which includes various object categories commonly encountered in autonomous driving scenarios.

## Getting Started
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/AutoPercept.git
    cd AutoPercept
    ```
2. **Setup Environment:**
    ```bash
   # Install required dependencies
    pip install -r requirements.txt
    ```

4. **Running the Gooey App:**
   ```bash
     python AutoPercept.py
    ```

5. **Specifying Pre-inference parameters**
    - NOTE : Trained Weights can be found in the "Model_Weights" directory
    - NOTE : Some example videos can be found to run inference on in the "Sample_Vids" directory
  
6. **Inference**
   - After specifying the pre-inference params, hit the "Start" Button to start the inference process
   - A new window will open up showing the detections being made for each frame of the video
   - After inference is completed, you will find the video saved in the 'output_path' directory

## Inference Examples

## Future Scope
- **Monocular Depth Estimation** : I aim to add functionality of simultaneous depth estimation and objection detection very soon
- **Simultaneous Localization and Mapping (SLAM)** : Due to the fact that the KITTI dataset also contains LiDAR and 3D Point Cloud data, It will be possible to add functionalities to visualize SLAM Processes in real time


