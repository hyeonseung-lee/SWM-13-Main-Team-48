# using YOLO
from unittest import result
import cv2
import time
from django.utils import timezone
from camera.models import *
from camera.views import *
# from django.core.files.base import ContentFile
import os
def default_streaming(cap):

    # For webcam input:
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@172.16.101.157:554/profile2/media.smp')
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@aicctv.iptime.org:554/profile2/media.smp')
    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@172.16.100.251:554/profile2/media.smp')
    
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue


        _, jpg = cv2.imencode('.jpg', image)
        frame = jpg.tobytes()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')

    # out.release()
    # output.release()
    cap.release()   
       