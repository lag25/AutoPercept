# AutoPercept
A project focused on YOLO object detection trained on the KITTI dataset. This repository serves as a starting point for implementing object detection capabilities in autonomous systems using YOLO architecture.

## Overview
AutoPercept leverages the power of YOLO (You Only Look Once) for real-time object detection and tracking. The model is trained on the KITTI dataset, enabling accurate detection of objects in various driving scenarios. The open sourced YOLOv8 weights from Ultralytics has been utilized for model training.

## Key Features
- **YOLO Object Detection:** Real-time detection and tracking of objects using YOLO architecture.
- **Real Time Counting:** Functionality for counting and displaying the amount of detections for each class has been added
- **Pythonic UI** : AutoPercept can also be used as a Wrapper or Interface to run inference on videos using custom model weights.
- **Saveable Results** : Functionality to save the video with the detections in a given directory with a given name has also been added
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
    ![image](https://github.com/lag25/AutoPercept/assets/116341862/694004f0-da32-4a91-a544-fb0fd4de4836)

    
    
    - NOTE : Trained Weights can be found in the "Model_Weights" directory
    - NOTE : Some example videos can be found to run inference on in the "Sample_Vids" directory
  
6. **Inference**
    ![image](https://github.com/lag25/AutoPercept/assets/116341862/1e13563f-3717-4774-a802-d97459bb86f9)

   - After specifying the pre-inference params, hit the "Start" Button to start the inference process
   - A new window will open up showing the detections being made for each frame of the video
   - After inference is completed, you will find the video saved in the 'output_path' directory

## Performance Metrics
1. Precision, Recall, Mean Average Precision@IoU=0.5 and Mean Average Precision@IoU=0.5-0.95 for each class over the validation dataset
![image](https://github.com/lag25/AutoPercept/assets/116341862/a99057b6-73df-4d3f-8e44-fae4197a5711)
2. Model Training Performance
    ![results](https://github.com/lag25/AutoPercept/assets/116341862/4a3964e9-4940-4f4c-b4b6-4cb70dd03f35)
## Working Examples
https://github.com/lag25/AutoPercept/assets/116341862/c158140f-758d-4827-90d0-e42010b9d60c

![readme_ex](https://github.com/lag25/AutoPercept/assets/116341862/e123094e-683e-4043-8ea3-c34cfeb64ab7)

## Future Scope
- **Monocular Depth Estimation** : I aim to add functionality of simultaneous depth estimation and objection detection very soon
- **Simultaneous Localization and Mapping (SLAM)** : Due to the fact that the KITTI dataset also contains LiDAR and 3D Point Cloud data, It will be possible to add functionalities to visualize SLAM Processes in real time


