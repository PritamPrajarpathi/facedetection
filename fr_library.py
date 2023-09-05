import cv2
import numpy as np
from PIL import Image
import os
import face_recognition

# Parent path for face image database
parent_path = 'Image'

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

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically

    # Convert the image from BGR (OpenCV) to RGB (face_recognition)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find face locations in the image
    face_locations = face_recognition.face_locations(rgb_img)

    # Iterate through each detected face and perform recognition
    for (top, right, bottom, left) in face_locations:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

        # Crop the face region for recognition
        face_image = rgb_img[top:bottom, left:right]

        # Predict the ID and confidence for the detected face
        face_encodings = face_recognition.face_encodings(face_image)
        if len(face_encodings) == 0:
            name = "unknown"
            confidence = "N/A"
        else:
            face_encoding = face_encodings[0]
            face_distances = face_recognition.face_distance(list(id_names.keys()), face_encoding)
            min_distance = min(face_distances)
            id = list(id_names.keys())[np.argmin(face_distances)]
            name = id_names[id]
            confidence = "{0}%".format(round((1 - min_distance) * 100, 2))

        # Display the name and confidence score on the screen
        cv2.putText(img, str(name), (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (left + 5, bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

    # Show the video frame with face recognition information
    cv2.imshow('camera', img)

    # Press 'ESC' key to exit the video stream
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Clean up and release the camera
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
