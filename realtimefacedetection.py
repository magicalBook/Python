# -*- coding: utf-8 -*-
from mtcnn.mtcnn import MTCNN
import cv2
import os

inputVideoPath = input('video path: ')

detector = MTCNN()


capture = cv2.VideoCapture(inputVideoPath)

#fps = int(round(capture.get(cv2.CAP_PROP_FPS)))

#width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

#fourcc = cv2.VideoWriter_fourcc(*'XVID')

while True:
    ret,frame = capture.read()
    if not ret:
        break
        
    frame = cv2.flip(frame,1)
    face = detector.detect_faces(frame)
    
    for i in range(len(face)):
        box = face[i]['box']
        keypoints = face[i]['keypoints']
        
        cv2.rectangle(frame,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,0,0),2)
        cv2.circle(frame,keypoints['left_eye'],1,(255,0,0),4)
        cv2.circle(frame,keypoints['right_eye'],1,(255,0,0),4)
        cv2.circle(frame,keypoints['nose'],1,(255,0,0),4)
        cv2.circle(frame,keypoints['mouth_left'],1,(255,0,0),4)
        cv2.circle(frame,keypoints['mouth_right'],1,(255,0,0),4)

    
    cv2.imshow('Real time face detection',frame)
    
    key = cv2.waitKey(3)
    if key == 27:
        break
        
capture.release()
cv2.destroyAllWindows()