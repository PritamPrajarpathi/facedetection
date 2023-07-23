import tkinter as tk
from tkinter import simpledialog
import subprocess
import os
import sys
import util
import cv2
import numpy as np
from PIL import Image

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'Recognize me', 'green', self.run_face_recognition)
        self.login_button_main_window.place(x=750, y=200)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'Register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('trainer/face_recognition.yml')
        cascadePath = "haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(cascadePath)
        self.id_names = {}
        self.load_id_names()

    def load_id_names(self):
        # Load ID names from the text file (id_names.txt)
        with open('trainer/id_names.txt', 'r') as file:
            for line in file:
                id_str, name = line.strip().split(': ')
                self.id_names[int(id_str)] = name

    def register_new_user(self):
        name = simpledialog.askstring("Input", "Enter the user name:")
        if name:
            print("Running face capture for user:", name)
            video = cv2.VideoCapture(0)
            facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            count = 0
            paths = "Image/" + name
            isExist = os.path.exists(paths)
            if isExist:
                print(f"{name} is already taken!")
                name = simpledialog.askstring("Input", "Enter a different user name:")
                if not name:
                    print("Name not entered. Face capture canceled.")
                    return
                paths = "Image/" + name
            else:
                os.makedirs(paths)

            while True:
                ret, frame = video.read()
                faces = facedetect.detectMultiScale(frame, 1.3, 5)
                for x, y, w, h in faces:
                    count = count + 1
                    image_path = f"./Image/{name}/{name}{count}.jpg"
                    print(f"Creating Image.............{image_path}")
                    cv2.imwrite(image_path, frame[y:y+h, x:x+w])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 3)
                cv2.imshow("WindowFrame", frame)
                cv2.waitKey(1)
                k = cv2.waitKey(30) & 0xff
                if k==27:
                    break
                if count>29:
                    break

                time.sleep(0.1)

            video.release()
            cv2.destroyAllWindows()

            # Run face_training.py using subprocess
            print("Running face training...")
            python_cmd = sys.executable
            training_process = subprocess.Popen([python_cmd, "face_training.py"], text=True)
            training_process.wait()
            print("Face training completed.")
        else:
            print("Name not entered. Face capture canceled.")

    def run_face_recognition(self):
        print("Running face recognition...")
        video = cv2.VideoCapture(0)
        video.set(3, 640)  # Set video width
        video.set(4, 480)  # Set video height

        # Define min window size to be recognized as a face
        minW = 0.1 * video.get(3)
        minH = 0.1 * video.get(4)

        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            ret, img = video.read()
            img = cv2.flip(img, 1)  # Flip vertically

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = self.faceCascade.detectMultiScale(gray,
                                                     scaleFactor=1.2,
                                                     minNeighbors=5,
                                                     minSize=(int(minW), int(minH)),
                                                     )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

                # Check if confidence is less than 100 ==> "0" is a perfect match
                if confidence < 100:
                    name = self.id_names[id]
                    confidence = "  {0}%".format(round(100 - confidence))
                else:
                    name = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

            # Convert the image to RGB format for tkinter display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert the RGB image to PIL format
            pil_img = Image.fromarray(img_rgb)
            # Convert the PIL image to PhotoImage for tkinter display
            tk_img = util.get_tk_image(pil_img)

            self.webcam_label.config(image=tk_img)
            self.webcam_label.image = tk_img  # Keep a reference to prevent garbage collection

            # Exit the loop when 'ESC' key is pressed
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = App()
    root = app.main_window
    root.title("Face Recognition GUI")
    root.mainloop()
