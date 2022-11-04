# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

from django.utils import timezone

import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    # parser.add_argument('config', help='test config file path')
    parser.add_argument('config', type=str, default='camera/ai_models/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py')
    parser.add_argument('checkpoint', type=str, default= 'camera/ai_models/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth', help='checkpoint file')
    parser.add_argument('label', type=str, default= "camera/ai_models/mmaction2/tools/data/kinetics/label_map_k400.txt", help='label file')
    parser.add_argument(
        # '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    parser.add_argument(
        '--average-size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    parser.add_argument(
        '--drawing-fps',
        type=int,
        default=20,
        help='Set upper bound FPS value of the output drawing')
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=4,
        help='Set upper bound FPS value of model inference')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    assert args.drawing_fps >= 0 and args.inference_fps >= 0, \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args


def webcam_stream():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    cur_time = time.time()
    
   
    while True:

        msg = 'Waiting for action ...'
        success, frame = camera.read()
        if success==False:
            print('error')
            continue
        frame_queue.append(np.array(frame[:, :, ::-1]))

        if len(result_queue) != 0:
            text_info = {}
            results = result_queue.popleft()
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                # print("frame1 :",frame)
                print("text1 :",text)
                print("location1 :",location)
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
        elif len(text_info) != 0:
            for location, text in text_info.items():
                print("text2 : ", text)
                print("location2 : ", location)
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)        
        else:
            # print("frame3 :",frame)
            print("msg3 :",msg)
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

           

        # ------------------------------------
        # now=timezone.now().strftime('%y/%m/%d - %H:%M:%S')
        # cv2.imwrite('camera/record_video/record_img/{}.jpg'.format(now),frame)
        # ------------------------------------

        # try:
        # _, jpeg = cv2.imencode('.jpg', frame)
        # jpegbytes = jpeg.tobytes()
        # print(jpegbytes)
        # if jpegbytes is None:
        #     print('hi')
        # yield(b'--frame\r\n'
            # b'Content-Type: image/jpeg\r\n\r\n' + jpegbytes + b'\r\n\r\n')    
        # except:
        #     continue
        # # ------------------------------------
        # cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def inference():
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = collate([cur_data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            cur_data = scatter(cur_data, [device])[0]

        with torch.no_grad():
            scores = model(return_loss=False, **cur_data)[0]

        score_cache.append(scores)
        scores_sum += scores

        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()

        if inference_fps > 0:
            # add a limiter for actual inference fps <= inference_fps
            sleep_time = 1 / inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    camera.release()
    cv2.destroyAllWindows()


def webcam_main():
    global frame_queue, camera, frame, results, threshold, sample_length, \
        data, test_pipeline, model, device, average_size, label, \
        result_queue, drawing_fps, inference_fps

    # args = parse_args()
    # average_size = args.average_size
    # threshold = args.threshold
    # drawing_fps = args.drawing_fps
    # inference_fps = args.inference_fps
    average_size = 1
    threshold = 0.01
    drawing_fps = 20
    inference_fps = 4

    # device = torch.device(args.device)
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    # cfg = Config.fromfile(args.config)
    # cfg.merge_from_dict(args.cfg_options)

    model = init_recognizer('camera/ai_models/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py', 'camera/ai_models/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth', device=device)
    # camera = cv2.VideoCapture(args.camera_id)
    camera = cv2.VideoCapture(0)
    # camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    data = dict(img_shape=None, modality='RGB', label=-1)

    # with open(args.label, 'r') as f:
    with open("camera/ai_models/mmaction2/tools/data/kinetics/label_map_k400.txt", 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    cfg = model.cfg
    sample_length = 0
    pipeline = cfg.data.test.pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    try:
        frame_queue = deque(maxlen=sample_length)
        result_queue = deque(maxlen=1)
        pw = Thread(target=webcam_stream, args=(), daemon=True)
        # pw = Thread(target=webcam_stream, args=())
        pr = Thread(target=inference, args=(), daemon=True)
        # pr = Thread(target=inference, args=())
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    webcam_main()




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
