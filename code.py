import cv2
from random import randrange


trained_face_data=cv2.CascadeClassifier('detection_face.xml')
trained_eye_data=cv2.CascadeClassifier('detection_eyes.xml')
video=cv2.VideoCapture(0)
while True :
    successful_frame,frame=video.read()
    grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinates=trained_face_data.detectMultiScale(grayscale)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),3)
        eyes_coordinates=trained_eye_data.detectMultiScale(grayscale)
        for (ex,ey,ew,eh) in eyes_coordinates:
            cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('detect_face',frame)
    cv2.waitKey(1)








