import tkinter as tk
from tkinter import simpledialog
import subprocess
import os
import sys

def run_face_capture():
    name = simpledialog.askstring("Input", "Enter the user name:")
    if name:
        print("Running face capture for user:", name)
        # Get the correct python command based on the OS
        python_cmd = sys.executable
        capture_process = subprocess.Popen([python_cmd, "face_capture.py"], stdin=subprocess.PIPE, text=True)
        capture_process.communicate(input=name)
    else:
        print("Name not entered. Face capture canceled.")

def run_face_training():
    print("Running face training...")
    python_cmd = sys.executable
    training_process = subprocess.Popen([python_cmd, "face_training.py"], text=True)
    training_process.wait()
    print("Face training completed.")

def run_face_recognition():
    print("Running face recognition...")
    python_cmd = sys.executable
    recognition_process = subprocess.Popen([python_cmd, "face_recognition_lbph.py"], text=True)
    recognition_process.wait()
    print("Face recognition completed.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Face Recognition GUI")

    btn_new = tk.Button(root, text="New", command=run_face_capture)
    btn_new.pack(pady=10)

    btn_train = tk.Button(root, text="Train", command=run_face_training)
    btn_train.pack(pady=5)

    btn_recognize = tk.Button(root, text="Recognize", command=run_face_recognition)
    btn_recognize.pack(pady=5)

    root.mainloop()
