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

def mmaction_stream():

    # For webcam input:
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@172.16.101.157:554/profile2/media.smp')
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@aicctv.iptime.org:554/profile2/media.smp')
    cap = cv2.VideoCapture(0)
    
    # out=0
    
    q=CircleQueue(300)
    temp=CircleQueue(600)
    ck=0
    while cap.isOpened():
        # if out==0:
        #     # ------------------------------------
        #     # 웹캠의 속성 값을 받아오기
        #     # 정수 형태로 변환하기 위해 round
        #     w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

        #     # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
        #     fourcc = cv2.VideoWriter_fourcc(*'H264')

        #     # print(w,h,fps,fourcc)
        #     # 웹캠으로 찰영한 영상을 저장하기
        #     # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
            
        #     now=timezone.now().strftime('%y%m%d_%H-%M-%S')
        #     out = cv2.VideoWriter('camera/record_video/{}.avi'.format(now), fourcc, fps, (w, h))
            
        #     # print(now)
        #     # ------------------------------------

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        
        now=timezone.now()
        t=now.strftime('%y%m%d_%H-%M-%S')
        cv2.imwrite('camera/record_video/record_img/{}.jpg'.format(t),image)
        # print(image)
        # out.write(image)
        
        _, jpg = cv2.imencode('.jpg', image)
        frame = jpg.tobytes()

        # key=cv2.waitKey(0)
        # if  key == 27: # esc
        #     print('hi')
        content = ContentFile(frame)
        img_model=Image()
        img_model.datetime=now
        img_model.image.save('{}.jpg'.format(t),content,save=False)
        img_model.save()

        yield(b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')

    # out.release()
    cap.release()   
        # # if 조건이 발생시 and ck<=0:
        # #   ck=q.count()
        # #   temp.enqueue(q.dequeue())

        # # elif 조건 발생시 and ck!=0:
        # #   temp.enqueue(q.dequeue())

        # #if ck>=0:
        # #   temp.enqueue(q.dequeue())
        # #   ck-=1
        
        # if q.is_full():
        #     q.dequeue()
        #     q.enqueue(image)

        # else:
        #     q.enqueue(image)


        # # q.printCQueue()
        
        # # if q.count()>
       



class CircleQueue:

    def __init__(self, size):
        self.MAX_SIZE = size
        self.queue = [None] * size
        self.head = -1
        self.tail = -1
    
    def count(self):
        if self.head>self.tail:
            return self.MAX_SIZE-(self.head-self.tail)
        else:
            return self.tail-self.head

    def is_full(self):
        if ((self.tail + 1) % self.MAX_SIZE == self.head):
            return True
        else:
            return False
    # 삽입
    def enqueue(self, data):

        if self.is_full():
            raise IndexError('Queue full')

        elif (self.head == -1):
            self.head = 0
            self.tail = 0
            self.queue[self.tail] = data

        else:
            self.tail = (self.tail + 1) % self.MAX_SIZE
            self.queue[self.tail] = data

    # 삭제
    def dequeue(self):
        if (self.head == -1):
            raise IndexError("The circular queue is empty\n")

        elif (self.head == self.tail):
            temp = self.queue[self.head]
            self.head = -1
            self.tail = -1
            return temp
        else:
            temp = self.queue[self.head]
            self.head = (self.head + 1) % self.MAX_SIZE
            return temp

    def printCQueue(self):
        if(self.head == -1):
            print("No element in the circular queue")

        elif (self.tail >= self.head):
            for i in range(self.head, self.tail + 1):
                print(self.queue[i], end=" ")
            print()
        else:
            for i in range(self.head, self.MAX_SIZE):
                print(self.queue[i], end=" ")
            for i in range(0, self.tail + 1):
                print(self.queue[i], end=" ")
            print()
