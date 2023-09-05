import os.path
import tkinter as tk
import util
import cv2
from PIL import Image, ImageTk
import subprocess
import datetime
import face_recognition
import os,sys
import time


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.title("Face Recognition App 5th year Group 1 project")
        self.main_window.geometry("1200x520+350+100")

        self.main_window.tk_setPalette(background="#fff")
        self.login_button_main_window = util.get_button(self.main_window, 'Check In', 'green', self.login)
        self.login_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)
        self.add_webcam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

    def add_webcam(self, label):
            if 'cap' not in self.__dict__:
                self.cap = cv2.VideoCapture(0)
            self._label = label
            self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if ret:
            # Detect faces in the frame
            self.most_recent_capture_arr = frame
            faces = self.detect_faces(frame)
            
            # Draw rectangles around detected faces
            frame_with_rectangles = self.draw_rectangles(frame, faces)
            
            # Display the frame with rectangles
            img_ = cv2.cvtColor(frame_with_rectangles, cv2.COLOR_BGR2RGB)
            self.most_recent_capture_pil = Image.fromarray(img_)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)
        
        self._label.after(20, self.process_webcam)
                
    def detect_faces(self, frame):
        # Load the pre-trained face detection model from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        return faces

    def draw_rectangles(self, frame, faces):
        frame_with_rectangles = frame.copy()
        for (x, y, w, h) in faces:
            # Draw a rectangle around each detected face
            cv2.rectangle(frame_with_rectangles, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return frame_with_rectangles

    def login(self):
        unknown_img_path = './.tmp.jpg'
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)
        output = str(subprocess.check_output(['face_recognition', self.db_dir, unknown_img_path]), encoding='utf-8')
        print("Output:", output)  # Add this line to see the full output

        names = output.split(',')[1].splitlines()  # Split the names by lines
        names = [name.strip() for name in names]  # Remove leading and trailing whitespaces from each name

        # Create a set to store unique matched names and exclude unwanted elements
        matched_names = set(name.rstrip() for name in names if name not in ["unknown_person", "no_persons_found", unknown_img_path])

        if not matched_names:
            util.msg_box("Ups..", "Unknown person. Please register a new person or try again.")
        else:
            welcome_message = ", ".join(matched_names)
            util.msg_box("Welcome to face recognition", "Welcome, {}".format(welcome_message))

            # Log each name and its corresponding datetime
            with open(self.log_path, 'a') as f:
                for name in matched_names:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write('{},{}\n'.format(name, now))
                
        os.remove(unknown_img_path)
    
    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        user_name = name.strip().title()
        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(user_name)),self.register_new_user_capture) 
        self.capture_images(user_name)
        util.msg_box("Success!","User was successfully captured")
        self.register_new_user_window.destroy()


    def capture_images(self, user_name):
        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        count = 0

        # Create a subfolder for the user's images
        user_image_dir = os.path.join('Image', user_name)
        os.makedirs(user_image_dir, exist_ok=True)

        while count < 30:
            ret, frame = video.read()
            faces = facedetect.detectMultiScale(frame, 1.3, 5)
            for x, y, w, h in faces:
                count = count + 1
                img_name = os.path.join(user_image_dir, f'{user_name}_{count}.jpg')
                print(f"Creating Image.............{img_name}")
                cv2.imwrite(img_name, frame[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imshow("WindowFrame", frame)
            cv2.waitKey(1)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            time.sleep(0.1)
        
        video.release()
        cv2.destroyAllWindows()
        self.register_new_user_window.destroy()
        self.main_window.destroy()
    def run_face_training():
        print("Running face training...")
        python_cmd = sys.executable
        cv2.face.LBPHFaceRecognizer_create()
        training_process = subprocess.Popen([python_cmd, "face_training.py"], text=True)
        training_process.wait()
        print("Face training completed.")

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()
    
    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()
