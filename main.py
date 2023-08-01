import cv2
import face_recognition
import os

# Load known face images and names from the Image folder
known_face_encodings = []
known_face_names = []
image_dir = 'Image'
cv2.face.LBPHFaceRecognizer_create()
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('jpg'):
            image_path = os.path.join(root, file)
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            # Check if any faces were detected in the image
            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.basename(root))
            else:
                pass

# Set up the webcam

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the current face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name if no match is found

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin()
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Calculate confidence as a percentage (the lower the distance, the higher the confidence)
        confidence = (1 - face_distances[best_match_index]) * 100
        name_with_confidence = f"{name} ({confidence:.2f}%)"

        # Draw rectangle around the face and display the name with confidence
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 225, 0), 3)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name_with_confidence, (left + 6, bottom - 6), font, 0.5, (0, 0, 200), 1)

    # Display the frame with rectangles and names
    cv2.imshow("TTU Face Recognition Project", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
