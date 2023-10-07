import cv2
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

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

print("\n [INFO] Loading and splitting the data...")
faces, ids, id_dict = getImagesAndLabels(parent_path)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.2, random_state=42)

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
recognizer.train(X_train, np.array(y_train))

# Save the model into trainer/ directory
trainer_dir = 'trainer/'
os.makedirs(trainer_dir, exist_ok=True)
recognizer.save(os.path.join(trainer_dir, 'face_recognition.yml'))

# Print the number of faces trained
print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))

# Save ID names to a text file
with open('trainer/id_names.txt', 'w') as file:
    for key, value in id_dict.items():
        file.write(f"{value}: {key}\n")



# Testing phase
print("\n [INFO] Testing the trained model.")
correct_predictions = 0
total_predictions = len(X_test)

for i in range(len(X_test)):
    label, confidence = recognizer.predict(X_test[i])
    predicted_name = [name for name, id in id_dict.items() if id == label][0]
    true_name = [name for name, id in id_dict.items() if id == y_test[i]][0]

    if label == y_test[i]:
        correct_predictions += 1

    confidence_percentage = round(100 - confidence, 2)
    print(f"Test {i + 1}: Predicted Name: {predicted_name}, True Name: {true_name}, Confidence: {confidence_percentage:.2f}%")

accuracy = (correct_predictions / total_predictions) * 100
print("\n [INFO] Testing completed.")
print(f"Accuracy: {accuracy:.2f}%")
