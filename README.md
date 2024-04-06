# AutoPercept
A project focused on Autonomous Vehicle Perception processes. This repository serves as a starting point for implementing object detection and depth estimation capabilities in autonomous systems using YOLO architecture and Vision Transformers.

## Overview
AutoPercept uses YOLO (You Only Look Once) for real-time object detection and tracking and a vision transformer called MiDaS for a Monocular Depth Estimation. The model is trained on the KITTI dataset, enabling accurate detection and depth estimation of objects in various driving scenarios. The open sourced YOLOv8 weights from Ultralytics have been utilized for object detection model training. Zero Shot Depth Estimation is done using MiDAS since the model has already been trained on the KITTI dataset (You can learn more about MiDaS at https://github.com/isl-org/MiDaS)

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


![readme_ex](https://github.com/lag25/AutoPercept/assets/116341862/7d8bd1b7-b24c-4181-9f67-88d3cbc52b21)


## Future Scope
- ~~**Monocular Depth Estimation** : I aim to add functionality of simultaneous depth estimation and objection detection very soon.~~ (_This functionality has been added_)
- **Quantization** : Currently, the MiDaS depth estimation model runs on 1-10 fps depending upon the resolution of the image. To makes the inference faster I am going to add 4 bit Quantization to the model. This will involve converting the model hyperparameters from their 32 bit floating point representation to a 4 bit one
- **Simultaneous Localization and Mapping (SLAM)** : Due to the fact that the KITTI dataset also contains LiDAR and 3D Point Cloud data, It will be possible to add functionalities to visualize SLAM Processes in real time.
- **Object Tracking and Projection** : Using Kalman Filter, I am currently working on creating methods to Track these object's movements and visualize them in real time.


