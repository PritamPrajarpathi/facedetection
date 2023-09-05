import os.path
import tkinter as tk
import util
import cv2
from PIL import Image, ImageTk
import subprocess
import datetime
import face_recognition
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
        cv2.face.LBPHFaceRecognizer_create()

        # # Load known faces from the ./db folder

        # self.known_face_encodings = []  # List to store known face encodings
        # self.known_face_names = []      # List to store corresponding names
        # image_dir = 'db'

        # for file in os.listdir(image_dir):
        #     if file.endswith('.jpg'):
        #         image_path = os.path.join(image_dir, file)
        #         face_name = os.path.splitext(file)[0]  # Extract the name from the filename
        #         image = face_recognition.load_image_file(image_path)
        #         face_encodings = face_recognition.face_encodings(image)

        #         # Check if any faces were detected in the image
        #         if len(face_encodings) > 0:
        #             face_encoding = face_encodings[0]
        #             self.known_face_encodings.append(face_encoding)
        #             self.known_face_names.append(face_name)

        # self.add_webcam(self.webcam_label)

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
                # Load known faces from the ./db folder
        known_face_encodings = []  # List to store known face encodings
        known_face_names = []  
        image_dir = 'db'

        for file in os.listdir(image_dir):
            if file.endswith('.jpg'):
                image_path = os.path.join(image_dir, file)
                face_name = os.path.splitext(file)[0]  # Extract the name from the filename
                image = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(image)

                # Check if any faces were detected in the image
                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(face_name)

        # self.add_webcam(self.webcam_label)
        if ret:
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(img_)
            face_encodings = face_recognition.face_encodings(img_, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Draw a green rectangle

                # Use face recognition to recognize the name
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)              
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            self.most_recent_capture_arr = frame
            self.most_recent_capture_pil = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            self._label.imgtk = imgtk
            self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)


    def login(self):
        unknown_img_path = './.tmp.jpg'
        cv2.imwrite(unknown_img_path, self.most_recent_capture_arr)
        output = str(subprocess.check_output(['face_recognition', self.db_dir, unknown_img_path]), encoding='utf-8')
        print("Output:", output)  # Add this line to see the full output

        names = output.split(',')[1].splitlines()  # Split the names by lines
        names = [name.strip() for name in names]  # Remove leading and trailing whitespaces from each name
        
        bounding_box_list=[]

        # Create a set to store unique matched names and exclude unwanted elements
        matched_names = set(name.rstrip() for name in names if name not in ["unknown_person", "no_persons_found", unknown_img_path])

        if not matched_names:
            util.msg_box("Ups..", "Unknown person. Please register a new person or try again.")
        else:
            for (top, right, bottom, left), name in zip(bounding_box_list, matched_names):
                cv2.rectangle(self.most_recent_capture_arr, (left, top), (right, bottom), (0, 255, 0), 2)  # Draw a green rectangle
                cv2.putText(self.most_recent_capture_arr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
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
        self.register_new_user_window.geometry("600x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Save', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=30, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=30, y=400)

        # self.capture_label = util.get_img_label(self.register_new_user_window)
        # self.capture_label.place(x=10, y=0, width=700, height=500)

        # self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=30, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput your name:')
        self.text_label_register_new_user.place(x=30, y=70)

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")
        cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)),self.register_new_user_capture)
        util.msg_box("Success!","User was successfully captured")
        
        # embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

        # file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        # pickle.dump(embeddings, file)

        # util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()


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
