import os
import cv2
import face_recognition

def load_training_data(training_dir):
    encodings = []
    names = []
    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                encodings.append(encoding)
                names.append(person_name)
    return encodings, names

def recognize_faces(encodings, names):
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                matched_indices = [i for i, match in enumerate(matches) if match]
                counts = {}
                for i in matched_indices:
                    name = names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            top, right, bottom, left = face_locations[0]
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

training_dir = "Image"
encodings, names = load_training_data(training_dir)
recognize_faces(encodings, names)
