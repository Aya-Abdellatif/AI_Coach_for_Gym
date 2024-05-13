# AI Wallt-Sit Gym Coach

This repository contains a computer vision model that can categorize Wall-Sit excersize into Correct and Incorrect. The model achieves high accuracy.

## Features

* Accuracy: The model achieves 98.9% accuracy on the test data
* Easy to Use: This repo comes with `inference.py` file which uses the camera and categorizes real-time video data, and also video data

## Dataset

The dataset used to train the model consists of 42 images and stored in `data` folder. Most of the data is from Google Images, the rest of the data were made by our teammate John.

**Important Note 1:** You have to open the directory in the terminal and execute the following commands<br>
**Important Note 2:** You have to install Python 3.10.6
# How to Use
## Installtion
To install the dependencies, run:
```bash
pip install -r requirements.txt
```
## Usage
To use your camera to categorize realtime video data of sign language, run:<br>
```
python inference.py
```
This will open a window showing camera, make the Wall-Sit excersize and it will automatically tell if you're doing it right or wrong.
If realtime camera is not available, you can use a video file included in the same director as `inference.py` and you can use it easily by executing the following command:<br>
```
python inference.py <filename.extension>
```
For example if the file name is `myVideo.mp4` the commands will look like this:<br>
```
python inference.py myVideo.mp4
```

# How it Works

## Training
All of the training process is in `train.ipynb`. First the data (collected before) is loaded into the memory. Each image is converted into 10 images by doing Data Augmentation (Flipping, Zooming, etc..), The images then go into mediapipe to detect the landmarks of the pose in each image. These landmarks are then reepresented as a single list for each image with shape (22, 2). Head landmarks and Z-axis are not taken into consideration. Then we initialized an adjacency list containing the landmark and the neighbouring landmarks (which are directly connected to this landmark), note that some of the landmarks are missing on purpose and some landmarks is assumed that it doesn't connect to anything directly and that's because head and arm landmarks will not affect the correctness of the Wall-Sit excersie. Then a depth-first-search is done on the adjacency list to identify what angles we can consider during the training process and the output is a list with the shape of (20, 3) for each image (20 angles with 3 keypoints each to measure the angle between them). Then each entry in this list is passed to a function to calculate the angle. Once we calculated the 20 angles, we do this for tthe rest of the images and then these angles are passed to the neural network.
