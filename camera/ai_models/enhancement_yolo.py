# using YOLO
# from asyncio.windows_events import INFINITE
from unittest import result
import cv2
import mediapipe as mp
import torch
import numpy
import time


def compareDistance(prev, pres):
    xDistance = abs(prev[0] - pres[0])
    yDistance = abs(prev[1] - pres[1])
    return int(xDistance + yDistance)


def enhancement_yolo_stream():
    # YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # since we are only intrested in detecting person
    model.classes = [0]
    # device = torch.device("cuda:0") #gpu용
    device = torch.device("cpu")
    model.to(device)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    pose_estimator = []
    pose_estimator_detected_object_loc = []

    # For webcam input:
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(
        'rtsp://admin:somateam23@172.16.101.157:554/profile2/media.smp')

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        yolo_results = model([image])
        yolo_objects_label, yolo_objects_informations = yolo_results.xyxyn[0][:, -1].cpu(
        ).numpy(), yolo_results.xyxyn[0][:, :-1].cpu().numpy()
        x_shape, y_shape = image.shape[1], image.shape[0]
        ##########################################
        # 처음엔 새로운 object를 감지하도록 코드를 짤 까 생각했지만, 새로운 object 수가 많지 않다면 mediapipe pose 모델이 충분히 object를 다시 연산해도 괜찮을 것이라고 판단
        # + mediapipe pose model이 예민하지 않은 편 (노력에 비해 큰 성능 차이를 얻지 못할 것 같음)

        # diff_prev_next_len = len(yolo_objects_label) - len(pose_estimator)
        # exist_new_object = False
        # if diff_prev_next_len > 0 : # 새로운 object가 있다
        #   diff_prev_next_len = 0
        #   exist_new_object = True
        ##########################################

        next_pose_estimator_detected_object_loc = [
            0 for _ in range(len(pose_estimator))]
        for object_id, (yolo_label, yolo_info) in enumerate(zip(yolo_objects_label, yolo_objects_informations)):
            xmin, ymin, xmax, ymax, conf = yolo_info
            if conf > 0.5:
                x1, y1, x2, y2 = int(
                    xmin*x_shape), int(ymin*y_shape), int(xmax*x_shape), int(ymax*y_shape)

                ##########################################
                # 미디어파이프의 성능을 높이기 위한 시도...
                # 효과가 있는 것 같음!
                margin = 10
                if y1 - margin > 0:
                    y1 = y1 - margin
                if y2 + margin < y_shape:
                    y2 = y2 + margin
                if x1 - margin > 0:
                    x1 = x1 - margin
                if x2 + margin < x_shape:
                    x2 = x2 + margin
                tmp_image = image[y1: y2, x1: x2]
                ##########################################

                mx, my = int((x1 + x2)/2), int((y1 + y2)/2)
                selected_object_index = 0
                if ((len(pose_estimator) == 0) or object_id == len(pose_estimator)):  # add new object
                    pose = mp_pose.Pose(
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)
                    pose_estimator.append(pose)
                    next_pose_estimator_detected_object_loc.append(
                        [mx, my])  # detected object's boundary save
                    selected_object_index = len(pose_estimator) - 1
                else:  # object's location update
                    previous_score = numpy.Infinity
                    for index, pre_info in enumerate(pose_estimator_detected_object_loc):
                        # print(f'pre info : {pre_info}, next info : {[mx, my]}')
                        if type(pre_info) == type(1):
                            continue
                        score = compareDistance(pre_info, [mx, my])
                        if (score < previous_score):
                            previous_score = score
                            selected_object_index = index
                    next_pose_estimator_detected_object_loc[selected_object_index] = [
                        mx, my]

                tmp_image = image[y1: y2, x1: x2]
                image.flags.writeable = False
                results = pose_estimator[selected_object_index].process(
                    tmp_image)

                image.flags.writeable = True
                mp_drawing.draw_landmarks(
                    tmp_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        pose_estimator_detected_object_loc = next_pose_estimator_detected_object_loc

        _, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # cv2.imshow('MediaPipe Pose with Yolo', cv2.flip(image, 1))
        # cv2.imwrite(f'save_images/test.png', background)
        # if cv2.waitKey(5) & 0xFF == 27:
        #   break
    cap.release()
