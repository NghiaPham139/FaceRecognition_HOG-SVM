import numpy as np
import os
import cv2
import dlib

hog_face_detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
path = "C:\\Users\\Nghia\\desktop\\data\\Data\\"# path were u want store the data set
id = raw_input('enter user name ')

try:
    # Create target Directory
    os.mkdir(path+str(id))
    print("Directory " , path+str(id),  " Created ")
except Exception:
    print("Directory " , path+str(id) ,  " already exists")
sampleN=0;

while 1:

    ret, img = cap.read()
    frame = img.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)

    for face in faces:

        sampleN=sampleN+1;
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        cv2.imwrite(path+str(id)+ "\\" +str(sampleN)+ ".jpg", gray[y:y+h, x:x+w])

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.waitKey(100)

    cv2.imshow('img',img)

    cv2.waitKey(1)

    if sampleN > 10:

        break

cap.release()

cv2.destroyAllWindows()