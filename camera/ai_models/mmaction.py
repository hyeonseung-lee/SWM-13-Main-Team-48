# using YOLO
from unittest import result
import cv2
import mediapipe as mp
import torch
import numpy
import time
from django.utils import timezone
from camera.models import *
from camera.views import *
from django.core.files.base import ContentFile
import os
def mmaction_stream():

    # For webcam input:
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@172.16.101.157:554/profile2/media.smp')
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@aicctv.iptime.org:554/profile2/media.smp')
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # 웹캠의 속성 값을 받아오기
        # 정수 형태로 변환하기 위해 round
        w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

        # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

        # print(w,h,fps,fourcc)
        # 웹캠으로 촬영한 영상을 저장하기
        # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
        ymd=timezone.now().strftime("%Y%m%d")
        now=timezone.now()
        

        output = cv2.VideoWriter(f'media/record_video/{ymd}/{now}.mp4', fourcc, 20, (w, h))
            
        output.write(image)

        _, jpg = cv2.imencode('.jpg', image)
        frame = jpg.tobytes()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')

    # out.release()
    output.release()
    cap.release()   
       