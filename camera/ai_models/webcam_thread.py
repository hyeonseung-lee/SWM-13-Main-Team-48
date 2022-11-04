# Copyright (c) OpenMMLab. All rights reserved.
# import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread
from django.utils import timezone
from camera.models import Video as Video_model
import cv2
import os
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

def show_results():
    print('Press "Esc", "q" or "Q" to exit')

    text_info = {}
    cur_time = time.time()
    cnt = 0
    while True:
        msg = 'Waiting for action ...'
        success, frame = camera.read()

        if success==False:
            print('error')
            continue
        
        if cnt == 3 : # 3번째 프레임만 저장. 20초동안 들어오는 프레임수 구한다음 -> 이중에서 25개만 얻도록 cnt 조건 맞추기
            frame_queue.append(np.array(frame[:, :, ::-1]))
            total_queue.append(frame)
            cnt = 0
        cnt += 1

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
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        elif len(text_info) != 0:
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

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
    global total_queue, top_reuslt
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    current_time = cur_time
    
    nomal = 0
    num_nomal = 2
    while True:
        cur_windows = []
        total_windows = []

        while len(cur_windows) == 0:
            # cnt += 1
            if len(frame_queue) == sample_length:
                cur_windows = list(np.array(frame_queue))
                total_windows = total_queue
                total_queue = []
                current_time = time.time()
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
            num_selected_labels = min(len(label), 3)

            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(
                scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]

            result_queue.append(results)
            scores_sum -= score_cache.popleft()
            # print(results)
            result_name=list(map(lambda x : x[0],results))

            top_reuslt=result_name[0]
            print(result_name)
            # if results[0][0] != 'fixing hair': # when abnomal behavior detect 이상행동일때
            if 'fixing hair' in result_name: # when abnomal behavior detect 이상행동일때
                print('이상행동임 ㅋㅋ')
                # make video info
                # add node in linked save list
                if nomal < num_nomal:
                    cur_info.time = current_time
                else:
                    cur_info.time = current_time
                    cur_info.video_name = current_time
                nomal = 0
                linked_video.add_node(Node(total_windows[:], cur_info))
            else: # when not abnomal 이상행동 아닐때
                print('이상아님 ㅋㅋ')

                # just save data then check count of abnomal
                if nomal <= num_nomal :
                    nomal += 1
                if nomal == num_nomal: # 앞의 영상 저장하기.
                    print('빈노드 넣을거임 ㅋㅋ')
                    linked_video.add_node(Node([], None))
                # 앞의 영상 저장됨. 이후로 비정상 행동 감지 X
        
        if inference_fps > 0:
            # add a limiter for actual inference fps <= inference_fps
            sleep_time = 1 / inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    camera.release()
    cv2.destroyAllWindows()

def save_video(request,save_path):
    # linked list
    # node 안에 있는 data 저장
    global top_reuslt
    images = []
    cur_name = None
    # fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    # H.264 코덱을 해야함
    # fourcc = cv2.VideoWriter_fourcc(*'h264') #오류나서 아래로 바꿔야함
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # h264 내부처리 조금다른거 빼고 같아서 별칭정도로 알면된다고 함
    while True:
        node = linked_video.popleft()

        if node != None :
            # 이상행동 - 이상행동 - ... - 정상행동 일 때, 비디오 저장
            if node.data_info is None and len(images) > 0:
                print('저장하러옴 ㅋㅋ')
                now=timezone.now()
                t=now.strftime('%y%m%d_%H-%M-%S')
                ymd=timezone.now().strftime("%Y%m%d")

                if not os.path.exists(f'media/record_video/{ymd}'):
                    os.makedirs(f'media/record_video/{ymd}')
                output = cv2.VideoWriter(f'media/record_video/{ymd}/{t}.mp4', fourcc, 10, (frame_width, frame_height))
                for image in images:
                    # print(image)
                    output.write(image)
                output.release()
                video_instance=Video_model.objects.create(
                    profile=request.user.profile,
                    video=save_path+'/{}/{}.mp4'.format(ymd,t),
                    type=top_reuslt,
                    datetime=now
                )
                images = []
                cur_name = None
            
            if node.data_info is not None:
                if cur_name is None:
                # 저장할 비디오 처음! images += node.data 그리고 cur_name = node.data_info_video_name
                    images = node.data
                    cur_name = node.data_info.video_name
                else:
                    images += node.data
 
        node = None

class Video :
    def __init__(self) -> None:
        self.time = time.time()
        self.video_name = time.time()

class Node:
    def __init__(self, data, info) -> None:
        self.data = data
        self.data_info = info
        self.next = None

class LinkedList:
    def __init__(self) -> None:
        self.head = None
        self.tail = None

    def add_node(self, node):
        if self.head is None:
            self.head = node
        elif self.tail is None:
            self.head.next = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node
    
    def popleft(self):
        node = self.head
        if node != None:
            self.head = node.next
            if self.head == self.tail:
                self.tail = None
            print('node의 개수 : ',len(node.data))
        return node
    

def webcam_thread_main(request):
    global frame_queue, camera, frame, results, threshold, sample_length, \
        data, test_pipeline, model, device, average_size, label, \
        result_queue, drawing_fps, inference_fps, total_queue, cur_info, cur_time, linked_video, frame_width, frame_height,top_reuslt
    top_reuslt='' #default로 빈문자열 담고 Predict될때 갱신

    average_size = 1
    threshold = 0.01
    drawing_fps = 20
    inference_fps = 4
    device = torch.device("cpu")

    save_path = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(save_path[:-7], 'media','record_video') 

    model = init_recognizer('camera/ai_models/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py', 'camera/ai_models/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth', device=device)
    camera = cv2.VideoCapture(0)

    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    data = dict(img_shape=None, modality='RGB', label=-1)

    # with open(args.label, 'r') as f:
    with open("camera/ai_models/mmaction2/tools/data/kinetics/label_map_k400.txt", 'r') as f:
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    total_queue = []
    cur_info = Video()
    linked_video = LinkedList()

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
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)
        sa = Thread(target=save_video, args=(request,save_path,), daemon=True)
        pw.start()
        pr.start()
        sa.start()
        pw.join()
    except KeyboardInterrupt:
        pass


# if __name__ == '__main__':
#     main()
