# from pathlib import Path
# import torch.backends.cudnn as cudnn
from numpy import random
from django.utils import timezone
from camera.ai_models.models.experimental import attempt_load
from camera.ai_models.utils.general import check_img_size, non_max_suppression, set_logging

# Copyright (c) OpenMMLab. All rights reserved.
import time
from collections import deque
from operator import itemgetter
from multiprocessing import Process, Value, Queue, Array

import cv2
import os
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (0, 0, 0)  # BGR, Black
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUED_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def count_people(frame:Queue, device, people_count:Value, countF_is_working):
    '''
    only detect people and count by yolov7
    when not exist people other operators(especially, inference :: expensive) not work    
    
    args:
    frame : help = webcam read frame, type = multiprocessing Queue
    device : help = torch.device(gpu_id or 'cpu')
    people_count : help = this function's result, type = multiprocessing Queue
    '''

    weights, imgsz ='./yolov7/yolov7.pt', 640
    set_logging()
    half = device.type != 'cpu'

    # load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    # cudnn.benchmark = True  # set True to speed up constant image size inference

    while True:
        print("count_people입니다")
        image = frame.get() # frame에 신호가 올때까지 여기에 머무름
        print("기다리는게 끝났나요?")

        current_time = time.time()
        img = [letterbox(np.array(x), imgsz, auto=True, stride=stride)[0] for x in [image]]
       
        # Stack
        img = np.stack(img, 0)
        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)
        
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad(): #필요한 메모리 줄어들고 연산속도 증가  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=0) # pred[0] length > 0 :: detect people 
        
        # send message "people_count var update" to show_results func
        people_count.value = len(pred[0]) #pred == [location, conf, classId], ...
        countF_is_working.value = 0
        print('yolo 걸리는 시간(초): ', time.time() - current_time)

def show_results(people_count:Value, to_count_func:Queue, to_inference_func:Queue, result_queue_index:Array, frame_width, frame_height, is_working, countF_is_working):
    '''
    frames : help = save frames for inferencing behaviors : type = multiprocessing Queue
    result_queue : help = save inferencing results : type = multiprocessing Queue
    '''
    print('Press "Esc", "q" or "Q" to exit')
    camera = cv2.VideoCapture(0)
    frame_width.value = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height.value = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    threshold = 0.01
    drawing_fps = 20
    sample_length = 25

    # label info
    with open("camera/ai_models/mmaction2/tools/data/kinetics/label_map_k400.txt", 'r') as f:
    # with open("mmaction2/tools/data/kinetics/label_map_k400.txt", 'r') as f:
        label = [line.strip() for line in f]

    text_info = {}
    cur_time = time.time()
    count = 0
    frame_queue = deque(maxlen=sample_length)
    total_frame_queue = []
    total_images = []
    normal = 0
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

    while is_working.value == 1: # wait until inference model loading is finished
        pass
    prev_time = cur_time = time.time()
    current_time = time.time()
    while True:
        print("show_result입니다")

        msg = 'Waiting for action ...'

        _, image = camera.read()

        # 1. send image(queue) to count_people func
        # 2. receive result(name = people_count, type = Value, cont = (-1:working, 0:not exist, >=1:exist)) from count_people func
         
        # 3. send images to inference
        # 4. receive results(Array)

        # 5. send total images to save
        if (people_count.value == 0 or countF_is_working.value == 0) and is_working.value == 0 and time.time() - prev_time > 2 : # count func not working and inference func not working and 작동 term 2초 이상
        # if people_count.value == 0 or countF_is_working.value == 0 : # count func not working
            # again
            to_count_func.put(image)
            prev_time = time.time()
            countF_is_working.value = 1 # count func working
        
        if people_count.value > 0 or len(frame_queue) > 0: # count_people func done and exist people | already gathering images(frame_queue) keep going
            # gather images(frame_queue)
            # send to inference
            first_enter = True
            if count == 3:
                count = 0
                total_frame_queue.append(image)
                frame_queue.append(np.array(image[:, :, ::-1]))
                if len(frame_queue) == sample_length and is_working.value == 0: # when frame_queue fulls | inference func not works
                    current_time = time.time()
                    is_working.value = 1
                    to_inference_func.put(frame_queue) # send images
                    frame_queue.clear() # deque 초기화 방법 알려주실 분.. (질문드리기) 왜 클리어를 사용하면 빈 리스트를...?

                    total_images += total_frame_queue.copy() # 저장할 이미지들
                    total_frame_queue.clear()
                    # 그냥 버릴 것인가 (deque maxlen에 의해 하나씩 빠짐)
            else:
                count += 1
        elif len(total_images) > 0: # not exist people long time, then save total images.
            # first time, save time.time()
            if first_enter:
                first_enter_time = time.time()
                first_enter = False
            
            elif time.time() - first_enter_time > 5:
                # save total images
                now=timezone.now()
                t=now.strftime('%y%m%d_%H-%M-%S')
                ymd=timezone.now().strftime("%Y%m%d")
                output = cv2.VideoWriter(f'media/record_video/{ymd}/{t}.mp4', fourcc, 10, (frame_width.value, frame_height.value))

                # output = cv2.VideoWriter(f'data/mysave/{int(time.time())}.mp4', fourcc, 10, (frame_width.value, frame_height.value))
                for img in total_images:
                    output.write(img)
                output.release()
                total_images = []
                first_enter = True 
            
        # show results
        # send total images to save func
        if result_queue_index[0] != result_queue_index[1]: # inference function already done
            # show and save
            text_info = {}
            results = [label[i] for i in result_queue_index]
            for i in range(5):
                result_queue_index[i] = 0 # reset

            exist_abnormal = False
            for i, result in enumerate(results):
                selected_label = result
                location = (0, 40 + i * 20)
                text = selected_label
                text_info[location] = text
                cv2.putText(image, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
                
            # save here
            # abnormal behaviors 묶어서 저장
            # normal (정상 판정 횟수)
                if result == 'fixing hair': # when detect abnormal
                    exist_abnormal = True
            
            normal_thres = 2
            
            if exist_abnormal:
                
                normal = 0
            else:
                if normal <= normal_thres:
                    normal += 1
                if normal == normal_thres:
                    now=timezone.now()
                    t=now.strftime('%y%m%d_%H-%M-%S')
                    ymd=timezone.now().strftime("%Y%m%d")
                    # send total images to save_video func
                    # to_save_func.put(total_images) # 처음 저장될 때 시간이 필요한 경우 int(initTime)
                    # output = cv2.VideoWriter(f'data/mysave/{int(time.time())}.mp4', fourcc, 10, (frame_width.value, frame_height.value))
                    output = cv2.VideoWriter(f'media/record_video/{ymd}/{t}.mp4', fourcc, 10, (frame_width.value, frame_height.value))
                    
                    for img in total_images:
                        output.write(img)
                    output.release()
                    total_images = []
                if normal > normal_thres:
                    total_images = []

        elif len(text_info) != 0: # 기존 inference 결과가 있으면
            if people_count.value == 0 and len(frame_queue)==0 and is_working.value == 0: # people not exist and inference not work
                text_info = {}
            else:
                for location, text in text_info.items():
                    cv2.putText(image, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

        else:
            msg = 'Waiting for action ...'

            cv2.putText(image, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        # success, jpg = cv2.imencode('.jpg', image)
        # if success:
        #     frame = jpg.tobytes()
        #     # if type(frame)==None:
        #     #     continue
        #     print(type(frame))
        #     yield(b'--frame\r\n'
        #             b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')
    
        
        cv2.imshow('camera', image)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()
    camera.release()
    cv2.destroyAllWindows()
        # success, jpg = cv2.imencode('.jpg', image)
        # print(success,jpg)
        # if :
        #     frame = jpg.tobytes()

        #     yield(b'--frame\r\n'
        #             b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')
    

def inference(device, from_show_func:Queue, result_queue_index:Queue, frame_width, frame_height, is_working):
    print("inference입니다")
    config = 'camera/ai_models/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
    checkpoint = 'camera/ai_models/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
    # config = 'mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
    # checkpoint = 'mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
    cfg_options = {}
    average_size = 1
    inference_fps = 4

    cfg = Config.fromfile(config)
    cfg.merge_from_dict(cfg_options)

    # model load
    model = init_recognizer(cfg, checkpoint, device=device)

    # data info
    data = dict(img_shape=None, modality='RGB', label=-1)

    cfg = model.cfg
    pipeline = cfg.data.test.pipeline
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'fixing hair' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUED_STEPS:
            # remove step to decode frames
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)

    assert sample_length > 0

    label_index = []
    is_working.value = 0
    while True:
        cur_windows = []

        while len(cur_windows) == 0:
            cur_windows = list(np.array(from_show_func.get()))
            current_time = time.time()
            if data['img_shape'] is None:
                data['img_shape'] = (frame_height.value, frame_width.value)

        cur_data = data.copy()
        cur_data['imgs'] = cur_windows
        cur_data = test_pipeline(cur_data)
        cur_data = collate([cur_data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            cur_data = scatter(cur_data, [device])[0]
        with torch.no_grad():
            scores = model(return_loss=False, **cur_data)[0]
        if len(label_index) < 1:
            label_index = [i for i in range(len(scores))]

        num_selected_labels = min(len(label_index), 5)
        scores_tuples = tuple(zip(label_index, scores))
        scores_sorted = sorted(
            scores_tuples, key=itemgetter(1), reverse=True)
        results = scores_sorted[:num_selected_labels]

        for i, (l, s) in enumerate(results): # 정보 전달에 실패한다면... 끔찍하군
            result_queue_index[i] = l
        is_working.value = 0
        print('inference 걸리는 시간(초): ',time.time() - current_time)

        if inference_fps > 0:
            # add a limiter for actual inference fps <= inference_fps
            sleep_time = 1 / inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()
        

def multiprocessing_main():
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    frame_width, frame_height = Value('i', 0), Value('i', 0)

    to_inference_func = Queue()
    to_count_func = Queue()

    people_count = Value('i', 0)
    result_queue_index = Array('i', [0,0,0,0,0])
    is_working = Value('i',0)
    countF_is_working = Value('i', 0)


    try:
        pw = Process(target=show_results, args=(people_count, to_count_func, to_inference_func, result_queue_index, frame_width, frame_height, is_working, countF_is_working))
        pr = Process(target=inference, args=(device, to_inference_func, result_queue_index, frame_width, frame_height, is_working))
        cp = Process(target=count_people, args=(to_count_func, device, people_count, countF_is_working))
        
        pw.start()
        pr.start()
        cp.start()
        pw.join()
    except KeyboardInterrupt:
        pass


# if __name__ == '__main__':
#     main()
