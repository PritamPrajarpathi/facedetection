import cv2
import os
import time

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haar_face.xml')
count = 0
nameID = str(input("Enter the user name : ")).lower()
paths= "Image/"+nameID
isExist = os.path.exists(paths)
if isExist :
    print(f"{nameID} is already taken!")
    nameID = str(input("Enter the user name : ")).lower()
else:
    os.makedirs(paths)

while True:

    ret,frame = video.read()
    faces= facedetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        count=count+1
        name='./Image/'+nameID+'/'+ nameID +str(count)+'.jpg'
        print(f"Creating Image.............{name}")
        cv2.imwrite(name,frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,225,0),3)
    cv2.imshow("WindowFrame",frame)
    cv2.waitKey(1)
    if count == 0:
        cv2.waitKey(5000)
    time.sleep(0.3)
      # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    if count>29:
        break
video.release()
cv2.destroyAllWindows()