import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sys import argv

model = tf.keras.models.load_model('98.9583.h5')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)



cap = cv.VideoCapture(0 if len(argv) != 2 else argv[1])

CONFIDENCE_THERSHOLD = 0.5

def calculate_angle(a, b, c, landmarks):
  a -= 11
  b -= 11
  c -= 11
  a = np.array(landmarks[a])
  b = np.array(landmarks[b])
  c = np.array(landmarks[c])
  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)
    
  if angle > 180.0:
      angle = 360 - angle
        
  return angle 

keypoints_to_consider = []
graph = {
    11: [12, 23],
    12: [11, 24],
    13: [],
    14: [],
    15: [],
    16: [],
    17: [],
    18: [],
    19: [],
    20: [],
    21: [],
    22: [],
    23: [11, 24, 25],
    24: [12, 23, 26],
    25: [23, 27],
    26: [24, 28],
    27: [25, 29, 31],
    28: [26, 30, 32],
    29: [27, 31],
    30: [28, 32],
    31: [27, 29],
    32: [28, 30]
}
def dfs(current):
    if len(current) == 3:
        if current[0] < current[2]:
            keypoints_to_consider.append(current.copy())
        return
    
    for node in graph[current[-1]]:
       dfs(current=current + [node])

def get_landmark_array(landmarks):
  lms = []
  result = []
  for l in landmarks:
    lms.append([l.x, l.y]) # 'z' will not be taken in consideration assuming that the image is a perfect or semi-perfect side view of a body, so z will not make a difference
  lms = np.array(lms[11:]) # Ignoring face landmarks
  for points in keypoints_to_consider:
    result.append(calculate_angle(*points, lms))
  return np.array(result)


def predict(img):
  imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  result = pose.process(imgRGB)
  if result.pose_landmarks:

    data = get_landmark_array(result.pose_landmarks.landmark)
    data = data.reshape((1, 20)) / 180.0
    result = model.predict(data)

    if float(result) >= CONFIDENCE_THERSHOLD: 
      cv.putText(img, "True" , (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
      cv.putText(img, "False" , (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

for i in range(11, 33):
    dfs([i])

while True:
    success, frame = cap.read()
    predict(frame)


    cv.imshow("Video", frame)
    cv.waitKey(1)
