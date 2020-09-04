#RUN COMMAND
#python driver_drowsiness_detector.py

#Importing libraries
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    #distance between vertical landmark
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    #distance between horizintal landmark
    C = dist.euclidean(eye[0], eye[3])
    # Getting average score
    ear = (A+B)/(2.0 * C)
    return ear

def mouth_aspect_ratio(inner):
    #Vertical mouth distance
    A = dist.euclidean(inner[1], inner[7])
    B = dist.euclidean(inner[2], inner[6])
    C = dist.euclidean(inner[3], inner[5])
    #Horizontal mouth dist
    D = dist.euclidean(inner[0], inner[4])
    #getting average score
    mar = (A + B + C)/(3.0 * D)
    return mar

#Setting threshold and readme
EYE_AR_THRESH = 0.2
MOUTH_AR_THRESH = 0.65
EYE_AR_CONSEC_FRAMES = 48
counter = 0
yawn_count = 0 
yawn_status = False


#Load the Face Landmark File and load algorithm 
path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

#Load the facial landmark points for left eye, right eye and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(iStart, iEnd) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]

vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    prev_yawn_status = yawn_status
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        inner = shape[iStart:iEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        mouthMAR = mouth_aspect_ratio(inner) 
        
        ear = (leftEAR+rightEAR)/2.0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        innerHull =  cv2.convexHull(inner)
        
        cv2.drawContours(frame, [leftEyeHull], -1,(0,255,0),1)
        cv2.drawContours(frame, [rightEyeHull], -1,(0,255,0),1)
        cv2.drawContours(frame, [innerHull], -1,(255,255,0),1)
        
        if mouthMAR>MOUTH_AR_THRESH:
            yawn_status = True
            cv2.putText(frame,"Yawning Alert!!!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255),2)
        else:
            yawn_status = False
        if prev_yawn_status == True and yawn_status == False:
            yawn_count += 1
        
        if ear< EYE_AR_THRESH or yawn_count>30:
            counter += 1
            if counter>=EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame,"Drowsiness Alert!!!",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255),2)
        else:
            COUNTER = 0
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Yawn: {:.2f}".format(yawn_count), (300, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)&0xFF
    if key== ord("q"):
        break
        
        
cv2.destroyAllWindows()
vs.stop()
