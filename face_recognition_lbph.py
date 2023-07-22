# face_recognition_lbph.py

import cv2
import numpy as np
from PIL import Image
import os

# Parent path for face image database
parent_path = 'Image'

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/face_recognition.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Load ID names from the text file (id_names.txt)
id_names = {}
with open('trainer/id_names.txt', 'r') as file:
    for line in file:
        id_str, name = line.strip().split(': ')
        id_names[int(id_str)] = name

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(int(minW), int(minH)),
                                         )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check if confidence is less than 100 ==> "0" is a perfect match
        if confidence < 100:
            name = id_names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            name = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
