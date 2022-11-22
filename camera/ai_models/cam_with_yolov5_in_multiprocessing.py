# import torch.backends.cudnn as cudnn
import django
django.setup()
# Copyright (c) OpenMMLab. All rights reserved.
import time
from collections import deque
from operator import itemgetter
from multiprocessing import Process, Value, Queue, Array
from django.utils import timezone
from pathlib import Path
from camera.models import Video as Video_model
from users.models import Profile
from camera.models import Camera
import os
import cv2
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose
# import multiprocessing as mp
# mp.set_start_method('fork')
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


def count_people(frame:Queue, device, people_count:Value, countF_is_working):
    '''
    only detect people and count by yolov7
    when not exist people other operators(especially, inference :: expensive) not work    
    
    args:
    frame : help = webcam read frame, type = multiprocessing Queue
    device : help = torch.device(gpu_id or 'cpu')
    people_count : help = this function's result, type = multiprocessing Queue
    '''

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes=[0]
    model.to(device)

    # Set Dataloader
    # cudnn.benchmark = True  # set True to speed up constant image size inference

    while True:
        image = frame.get() # frame에 신호가 올때까지 여기에 머무름
        current_time = time.time()
        yolo_results = model([image])
        yolo_objects_labels, yolo_objects_informations = yolo_results.xyxyn[0][:, -1].cpu().numpy(), yolo_results.xyxyn[0][:, :-1].cpu().numpy()

        # send message "people_count var update" to show_results func
        people_count.value = len(yolo_objects_labels)
        countF_is_working.value = 0
        print('yolo 걸리는 시간(초): ', time.time() - current_time)

def show_results(people_count:Value, to_count_func:Queue, to_inference_func:Queue, result_queue_index:Array, frame_width, frame_height, is_working, countF_is_working,to_stream:Queue,request_val,default_cam):
# def show_results(people_count:Value, to_count_func:Queue, to_inference_func:Queue, result_queue_index:Array, frame_width, frame_height, is_working, countF_is_working,to_stream:Queue):

    '''
    frames : help = save frames for inferencing behaviors : type = multiprocessing Queue
    result_queue : help = save inferencing results : type = multiprocessing Queue
    '''
    
    save_path = os.path.dirname(os.path.dirname(__file__))
    video_path = os.path.join(save_path[:-7], 'media','record_video') 
    thumbnail_path=os.path.join(save_path[:-7], 'media','record_img') 

    print('Press "Esc", "q" or "Q" to exit')
    camera = cv2.VideoCapture('rtsp://admin:somateam23@172.16.101.140:554/profile2/media.smp')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame_width.value = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height.value = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    threshold = 0.01
    # drawing_fps = 20
    sample_length = 25

    # label info
    with open("camera/ai_models/mmaction2/tools/data/kinetics/label_map_k400.txt", 'r') as f:
    # with open("tools/data/kinetics/label_map_k400.txt", 'r') as f:
        label = [line.strip() for line in f]

    text_info = {}
    count = 0
    frame_queue = deque(maxlen=sample_length)
    total_frame_queue = []
    total_images = []
    first_enter = True
    normal = normal_thres=2
    fourcc = cv2.VideoWriter_fourcc(*'avc1') # h264 내부처리 조금다른거 빼고 같아서 별칭정도로 알면된다고 함

    while is_working.value == 1: # wait until inference model loading is finished
        pass

    prev_time = cur_time = time.time()
    current_time = time.time()
    while True:

        _, image = camera.read()

        # 1. send image(queue) to count_people func
        # 2. receive result(name = people_count, type = Value, cont = (-1:working, 0:not exist, >=1:exist)) from count_people func
         
        # 3. send images to inference
        # 4. receive results(Array)

        # 5. send total images to save
        to_stream.put(image)
        if (people_count.value == 0 or countF_is_working.value == 0) and is_working.value == 0 and time.time() - prev_time > 2 : # count func not working and inference func not working and 작동 term 2초 이상
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
                frame_queue.append(np.array(image[:, :, ::-1])) # BGR -> RGB
                if len(frame_queue) == sample_length and is_working.value == 0: # when frame_queue fulls | inference func not works
                # 보낼 것인가
                    print('frame 25개 모으는 데 걸린 시간: ', time.time() - current_time)
                    current_time = time.time()
                    is_working.value = 1
                    to_inference_func.put(frame_queue.copy()) # send images
                    frame_queue.clear() # deque 초기화 방법 알려주실 분.. (질문드리기) 왜 클리어를 사용하면 빈 리스트를...?
                    # frame_queue = deque(maxlen=sample_length)
                    total_images += total_frame_queue.copy() # 저장할 이미지들
                    total_frame_queue.clear()
                    # total_frame_queue = [] # reset
                # 그냥 버릴 것인가 (deque maxlen에 의해 하나씩 빠짐)
            else:
                count += 1

        elif len(total_images) > 0 and normal<normal_thres-1: # not exist people long time, then save total images.
            # first time, save time.time()
            if first_enter:
                first_enter_time = time.time()
                first_enter = False
            
            elif time.time() - first_enter_time > 5:
                # save total images
                now=timezone.now()
                t=now.strftime('%y%m%d_%H-%M-%S')
                ymd=timezone.now().strftime("%Y%m%d")
                if not os.path.exists(f'media/record_video/{ymd}'):
                    os.makedirs(f'media/record_video/{ymd}')
                if not os.path.exists(f'media/record_img/{ymd}'):
                    os.makedirs(f'media/record_img/{ymd}')
                        
                output = cv2.VideoWriter(f'media/record_video/{ymd}/{t}.mp4', fourcc, 10, (frame_width.value, frame_height.value))

                thumbnail=True
                for img in total_images:
                    if thumbnail:
                        print('사진 저장')
                        cv2.imwrite(f'media/record_img/{ymd}/{t}.jpg',img)
                        thumbnail=False
                    # print(image)
                    output.write(img)
                
                output.release()
                request_value=request_val.value
                default_camera=default_cam.value
                # print(request_val)
                # print(type(request_val))
                # print(default_cam)
                # print(type(default_cam))

                profile=Profile.objects.get(id=request_value)
                default_cam_obj=Camera.objects.get(id=default_camera)
                video_instance=Video_model.objects.create(
                    profile=profile,
                    video=video_path+'/{}/{}.mp4'.format(ymd,t),
                    camera=default_cam_obj,
                    type=label[result_queue_index[0]],
                    datetime=now,
                    thumbnail=thumbnail_path+'/{}/{}.jpg'.format(ymd,t)
                )
                print(video_instance)
                # output = cv2.VideoWriter(f'data/mysave/{int(time.time())}.mp4', fourcc, 10, (frame_width.value, frame_height.value))
                
                # send_to_firebase_cloud_messaging_User(request.user.profile.fcm_toekn,request.user,top_reuslt)

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
                if result in 'fixing hair': # when detect abnormal
                    exist_abnormal = True
            
            if exist_abnormal:
                normal = 0
            else:
                if normal < normal_thres:
                    normal += 1
                if normal == normal_thres:
                    # send total images to save_video func
                    now=timezone.now()
                    t=now.strftime('%y%m%d_%H-%M-%S')
                    ymd=timezone.now().strftime("%Y%m%d")
                    if not os.path.exists(f'media/record_video/{ymd}'):
                        os.makedirs(f'media/record_video/{ymd}')
                    if not os.path.exists(f'media/record_img/{ymd}'):
                        os.makedirs(f'media/record_img/{ymd}')

                    output = cv2.VideoWriter(f'media/record_video/{ymd}/{t}.mp4', fourcc, 10, (frame_width.value, frame_height.value))
                
                    thumbnail=True

                    for img in total_images:
                        if thumbnail:
                            print('사진 저장')
                            cv2.imwrite(f'media/record_img/{ymd}/{t}.jpg',img)
                            thumbnail=False
                        # print(image)
                        output.write(img)
                    output.release()

                    request_value=request_val.value
                    default_camera=default_cam.value
                    # print(request_val)
                    # print(type(request_val))
                    # print(default_cam)
                    # print(type(default_cam))
                    

                    profile=Profile.objects.get(id=request_value)
                    default_cam_obj=Camera.objects.get(id=default_camera)
                    video_instance=Video_model.objects.create(
                        profile=profile,
                        video=video_path+'/{}/{}.mp4'.format(ymd,t),
                        camera=default_cam_obj,
                        type=label[result_queue_index[0]],
                        datetime=now,
                        thumbnail=thumbnail_path+'/{}/{}.jpg'.format(ymd,t)
                    )
                    print(video_instance)
                    
                    # send_to_firebase_cloud_messaging_User(request.user.profile.fcm_toekn,request.user,top_reuslt)

     
                    total_images = []
                else:
                    print('저장할게 있는 상황에서 이런경우가 있나?')
                    total_images = []

        elif len(text_info) != 0: # 기존 inference 결과가 있으면
            # if people_count.value == 0 and is_working.value == 0: # people not exist and inference not work
            if people_count.value == 0 and len(frame_queue) == 0 and is_working.value == 0: # people not exist and, gathering images and inference func not work
                text_info = {}
            else:
                for location, text in text_info.items():
                    cv2.putText(image, text, location, FONTFACE, FONTSCALE,
                                FONTCOLOR, THICKNESS, LINETYPE)

        else:
            msg = 'Waiting for action ...'
            cv2.putText(image, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        # if drawing_fps > 0:
        #     # add a limiter for actual drawing fps <= drawing_fps
        #     sleep_time = 1 / drawing_fps - (time.time() - cur_time)
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)
        #     cur_time = time.time()
    
    camera.release()
    cv2.destroyAllWindows()
    

def inference(device, from_show_func:Queue, result_queue_index:Queue, frame_width, frame_height, is_working):
    config = 'camera/ai_models/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
    # config = 'configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py'
    checkpoint = 'camera/ai_models/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
    # checkpoint = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'
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

        for i, (l, s) in enumerate(results):
            result_queue_index[i] = l
        
        is_working.value = 0
        print('inference 걸리는 시간(초): ',time.time() - current_time)

def multiprocessing_main(request,default_camera):
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    frame_width, frame_height = Value('i', 0), Value('i', 0)

    to_inference_func = Queue()
    to_count_func = Queue()
    to_stream=Queue()

    people_count = Value('i', 0)
    result_queue_index = Array('i', [0,0,0,0,0])
    is_working = Value('i',1)
    countF_is_working = Value('i', 0)

    # print(request.user.profile.id)
    # print(type(request.user.profile.id))
    # print(default_camera.id)
    # print(type(default_camera.id))

    request_val=Value('i',request.user.profile.id)
    default_cam=Value('i',default_camera.id)

    try:
        pw = Process(target=show_results, args=(people_count, to_count_func, to_inference_func, result_queue_index, frame_width, frame_height, is_working, countF_is_working,to_stream,request_val,default_cam), daemon=True)
        # pw = Process(target=show_results, args=(people_count, to_count_func, to_inference_func, result_queue_index, frame_width, frame_height, is_working, countF_is_working,to_stream), daemon=True)
        pr = Process(target=inference, args=(device, to_inference_func, result_queue_index, frame_width, frame_height, is_working), daemon=True)
        cp = Process(target=count_people, args=(to_count_func, device, people_count, countF_is_working), daemon=True)
        
        pw.start()
        pr.start()
        cp.start()
        want_break=False
        while True:
            image=to_stream.get() 
            success, jpg = cv2.imencode('.jpg', image)
            if want_break:
                break
            if success:
                frame = jpg.tobytes()
                yield(b'--frame\r\n'
                        b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')
        pw.join()
        
    except KeyboardInterrupt:
        pass


# if __name__ == '__main__':
#     main()
