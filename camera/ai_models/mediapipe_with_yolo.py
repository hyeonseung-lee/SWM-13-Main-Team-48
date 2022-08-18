#using YOLO
from unittest import result
import cv2
import mediapipe as mp
import torch
import numpy
import time

def mediapipe_with_yolo_stream():
  #YOLOv5
  model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
  #since we are only intrested in detecting person
  model.classes=[0]

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_pose = mp.solutions.pose

  pose_estimator = []
  pose_estimator_dim = []

  # For webcam input:
  # cap = cv2.VideoCapture('rtsp://admin:somateam23@172.16.101.157:554/profile2/media.smp')
  # cap = cv2.VideoCapture('rtsp://admin:somateam23@aicctv.iptime.org:554/profile2/media.smp')
  cap = cv2.VideoCapture(0)

  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      yoresults = model([image])
      yoobjects_label, yoobjects_informations = yoresults.xyxyn[0][:, -1].cpu().numpy(), yoresults.xyxyn[0][:, :-1].cpu().numpy()
      
      for yolabel, yoinfo in zip(yoobjects_label, yoobjects_informations):
        xmin, ymin, xmax, ymax, conf = yoinfo
        if conf > 0.5 :
          x_shape, y_shape = image.shape[1], image.shape[0]
          x1, y1, x2, y2 = int(xmin*x_shape), int(ymin*y_shape), int(xmax*x_shape), int(ymax*y_shape)
          cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

          # cv2.putText(image, f'{model.names[int(yolabel)]} : {conf}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)


          tmp_image = image[y1 : y2 , x1 : x2]
          results = pose.process(tmp_image)


          mp_drawing.draw_landmarks(
            tmp_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
          
      
      _, jpeg = cv2.imencode('.jpg', image)
      frame=jpeg.tobytes()

      yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
      
      # cv2.imshow('MediaPipe Pose with Yolo', cv2.flip(image, 1))
      # cv2.imwrite(f'save_images/test.png', background)
      # if cv2.waitKey(5) & 0xFF == 27:
      #   break
  cap.release()