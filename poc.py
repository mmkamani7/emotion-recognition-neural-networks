# Proof-of-concept
import cv2
import sys
from constants import *
from emotion_recognition import EmotionRecognition
from emotion_recognition import fileType, path, frames
import numpy as np
import json
# import argparse
import glob

cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

def brighten(data,b):
     datab = data * b
     return datab    

def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
  faces = cascade_classifier.detectMultiScale(
      image,
      scaleFactor = 1.1,
      minNeighbors = 4
  )
  # None is we don't found an image
  if not len(faces) > 0:
    return None
  max_area_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
      max_area_face = face
  # Chop image to face
  face = max_area_face
  image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
  except Exception:
    print("[+] Problem during resize")
    return None
  # cv2.imshow("Lol", image)
  # cv2.waitKey(0)
  return image

def load_image():
  for file in glob.glob(fileType):
    frame = cv2.imread(file,0)
    count = 0
    data = {}

    result = network.predict(frame)
    # cv2.imshow('frame', frame)

    if result is not None:
      for index, emotion in enumerate(EMOTIONS):
        cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
        cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

      face_image = feelings_faces[result[0].tolist().index(max(result[0]))]

      # for c in range(0, 3):
        # frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)

      cv2.imwrite("image_frame_%d.jpg" % count, frame)
      for index, emotion in enumerate(EMOTIONS):
        data.update({"frame":count})
        data.update({emotion:(130 + int(result[0][index] * 100))})
      with open('image_results.json', 'a') as obj:
        json.dump(data, obj)
        obj.write('\n')

      #plays the video for the user to see, although the video will be gray
      # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      # cv2.imshow('frame',gray)

def load_video():
    #change file name (.mp4 files work)
  for file in glob.glob(fileType + "*.mp4"):
    cap = cv2.VideoCapture(file)
    #cap = cv2.VideoCapture('WIN_20180308_18_33_08_Pro.mp4')
    count = 0
    data = {}

    while(cap.isOpened()):
      ret, frame = cap.read()
      result = network.predict(format_image(frame))
      # cv2.imshow('frame', frame)

      if result is not None:
        for index, emotion in enumerate(EMOTIONS):
          cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
          cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)

        face_image = feelings_faces[result[0].tolist().index(max(result[0]))]

        for c in range(0, 3):
          frame[200:320, 10:130, c] = face_image[:,:,c] * (face_image[:, :, 3] / 255.0) +  frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)

          if count % int(frames) == 0:
            cv2.imwrite("frame_%d.jpg" % count, frame)
            for index, emotion in enumerate(EMOTIONS):
              data.update({"frame":count})
              data.update({emotion:(130 + int(result[0][index] * 100))})
            with open('results.json', 'a') as obj:
              json.dump(data, obj)
              obj.write('\n')
          count += 1

        #plays the video for the user to see, although the video will be gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

# parser = argparse.ArgumentParser(description="Input the file type, path, and frames if applicable")

# parser.add_argument('fileType', help='Enter the word video or image')
# parser.add_argument('path', help='Enter the path to the directory you want to use')
# parser.add_argument('frames', help='For every number of frames, a frame will be saved. Enter the number you want - enter any number if you are using images instead of videos')

# args = parser.parse_args()

# Load Model
network = EmotionRecognition()
network.build_network()

#change file name (.mp4 files work)
#video_capture = cv2.VideoCapture('WIN_20180308_18_33_08_Pro.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX

feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
  feelings_faces.append(cv2.imread('./emojis/' + emotion + '.png', -1))

if fileType == "video":
  load_video()
elif fileType == "image":
  load_image()