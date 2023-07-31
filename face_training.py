# face_training.py

import cv2
import numpy as np
from PIL import Image
import os

# Parent path for face image database
parent_path = 'Image'

recognizer = cv2.face.LBPHFaceRecognizer_create() 
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(parent_path):
    faceSamples = []
    ids = []
    id_dict = {}

    for subdir in os.listdir(parent_path):
        dir_path = os.path.join(parent_path, subdir)
        if not os.path.isdir(dir_path):
            continue

        id_str = subdir.replace('.jpg', '')  # Remove '.jpg' from the subdirectory name

        if id_str not in id_dict:
            id_dict[id_str] = len(id_dict) + 1

        id = id_dict[id_str]  # Use the dictionary value as the ID

        for filename in os.listdir(dir_path):
            if not filename.endswith('.jpg'):
                continue

            imagePath = os.path.join(dir_path, filename)
            PIL_img = Image.open(imagePath).convert('L')  # Convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

    return faceSamples, ids, id_dict

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids, id_dict = getImagesAndLabels(parent_path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/ directory
trainer_dir = 'trainer/'
os.makedirs(trainer_dir, exist_ok=True)
recognizer.save(os.path.join(trainer_dir, 'face_recognition.yml'))

# Print the number of faces trained and end the program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

# Save ID names to a text file
with open('trainer/id_names.txt', 'w') as file:
    for key, value in id_dict.items():
        file.write(f"{value}: {key}\n")
