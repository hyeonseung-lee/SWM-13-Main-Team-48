# using YOLO
from unittest import result
import cv2
import mediapipe as mp
import torch
import numpy
import time
from django.utils import timezone
              
from camera.views import *

def mediapipe_with_yolo_stream():
    # YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # since we are only intrested in detecting person
    model.classes = [0]

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    pose_estimator = []
    pose_estimator_dim = []

    # For webcam input:
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@172.16.101.157:554/profile2/media.smp')
    # cap = cv2.VideoCapture('rtsp://admin:somateam23@aicctv.iptime.org:554/profile2/media.smp')
    cap = cv2.VideoCapture(0)
    
    out=0
    
    q=CircleQueue(300)
    temp=CircleQueue(600)
    ck=0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            if out==0:
                # ------------------------------------
                # 웹캠의 속성 값을 받아오기
                # 정수 형태로 변환하기 위해 round
                w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

                # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
                fourcc = cv2.VideoWriter_fourcc(*'H264')

                # print(w,h,fps,fourcc)
                # 웹캠으로 찰영한 영상을 저장하기
                # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
                now=timezone.now().strftime('%y/%m/%d - %H:%M:%S')
                # print(now)
                out = cv2.VideoWriter('camera/record_video/{0}.avi'.format(now), fourcc, fps, (w, h))
                # ------------------------------------
                # print(out)
            # print(out)

            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            
            out.write(image)

            yoresults = model([image])
            yoobjects_label, yoobjects_informations = yoresults.xyxyn[0][:, -1].cpu(
            ).numpy(), yoresults.xyxyn[0][:, :-1].cpu().numpy()

            for yolabel, yoinfo in zip(yoobjects_label, yoobjects_informations):
                xmin, ymin, xmax, ymax, conf = yoinfo
                if conf > 0.5:
                    x_shape, y_shape = image.shape[1], image.shape[0]
                    x1, y1, x2, y2 = int(
                        xmin*x_shape), int(ymin*y_shape), int(xmax*x_shape), int(ymax*y_shape)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # cv2.putText(image, f'{model.names[int(yolabel)]} : {conf}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                    tmp_image = image[y1: y2, x1: x2]
                    results = pose.process(tmp_image)

                    mp_drawing.draw_landmarks(
                        tmp_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


            # if 조건이 발생시 and ck<=0:
            #   ck=q.count()
            #   temp.enqueue(q.dequeue())

            # elif 조건 발생시 and ck!=0:
            #   temp.enqueue(q.dequeue())

            #if ck>=0:
            #   temp.enqueue(q.dequeue())
            #   ck-=1
            
            if q.is_full():
                q.dequeue()
                q.enqueue(image)

            else:
                q.enqueue(image)


            # q.printCQueue()
            
            # if q.count()>
            _, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            

            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            # cv2.imshow('MediaPipe Pose with Yolo', cv2.flip(image, 1))
            # cv2.imwrite(f'save_images/test.png', background)
            # if cv2.waitKey(5) & 0xFF == 27:
            #   break
        out.release()
    cap.release()



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
