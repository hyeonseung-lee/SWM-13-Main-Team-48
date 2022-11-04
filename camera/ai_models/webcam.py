# Copyright (c) OpenMMLab. All rights reserved.
# import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread
import multiprocessing as mp
from django.core.files.base import ContentFile
from django.utils import timezone
# from camera.models import *
from collections import deque
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


def webcam_stream():
    # print('Press "Esc", "q" or "Q" to exit')
    text_info = {}
    cur_time = time.time()
    while True:
    
        msg = 'Waiting for action ...'
        success, frame = camera.read()

        if success==False:
            print('error')
            continue
        frame_queue.append(np.array(frame[:, :, ::-1]))
        prev_queue.append(np.array(frame[:, :, ::-1]))
        # _, jpeg = cv2.imencode('.jpg', frame)
        # jpegbytes = jpeg.tobytes()

        #계속 담을거    
        # prev_queue.append(jpegbytes)

        if len(result_queue) != 0:

            ck=time.time()
            count=1

            text_info = {}
            results = result_queue.popleft()
            # print(results[0][0])
            for i, result in enumerate(results):
                selected_label, score = result
                if score < threshold:
                    break
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                # print("frame1 :",jpegbytes)
                # print("text1 :",text)
                # print("location1 :",location)
                
                # if 'cracking neck' in text:

                #     # store_mem_thread=Thread(target=sub_ins.store_mem(sub_maxlen),daemon=True)
                #     # store_mem_thread.start()
                #     # store_mem_thread.join()

                #     # now=timezone.now()
                #     # t=now.strftime('%y%m%d_%H-%M-%S')
                #     # cv2.imwrite('camera/record_video/record_img/{}.jpg'.format(t),frame)
                #     print(type(jpegbytes))
                #     content = ContentFile(jpegbytes)
                #     img_model=Image()
                #     img_model.datetime=now
                #     img_model.image.save('{}.jpg'.format(t),content,save=False)
                #     img_model.save()
                #     print('recording1')

                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
                
        elif len(text_info) != 0:
          
            count+=1
            if time.time()-ck>=1:
                # print(count)
                pass
               
            
            for location, text in text_info.items():

                # print("frame2 :",jpegbytes)
                # print("text2 : ", text)
                # print("location2 : ", location)
                # print(type(text))
                # if 'cracking neck' in text:
                #     # test_now = time.time()
                    
                #     # if get_frame==True and len(sub_queue)==sub_maxlen: #서브가 꽉차면 이제 멈추고 내보내기
                #     #     print(sub_queue.popleft())
                #     #     get_frame=False
                #     # elif get_frame==True : #서브 안찼고 아직 받아올때(서브가 찰때까지)
                #     #     sub_queue.append(prev_queue.popleft())
                #     #     print(len(sub_queue))

                #     # elif get_frame==False: # 서브 꽉찼고 이제 내보내기
                #     #     print(sub_queue.popleft())

                #     now=timezone.now()
                #     t=now.strftime('%y%m%d_%H-%M-%S')
                #     # cv2.imwrite('camera/record_video/record_img/{}.jpg'.format(t),frame)
                #     content = ContentFile(jpegbytes)
                #     print(type(jpegbytes))

                #     img_model=Image()
                #     img_model.datetime=now
                #     img_model.image.save('{}.jpg'.format(t),content,save=False)
                #     img_model.save()
                #     print('recording2')

                    # print(time.time() - test_now)
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)     
        else:
            # print("frame3 :",frame)
            # print("msg3 :",msg)
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)



        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                # print('streaming')
                # print(sleep_time)
            cur_time = time.time()


def inference():
    global prev_queue

    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    while True:

        cur_windows = []

        while len(cur_windows) == 0: # 한번 모델결과를 낼때마다 0이더라 -> 모델 한번돌면 inference 반복문이 다시 돌아서 초기화됨
            
            if len(frame_queue) == sample_length:
                #프레임정보인거같음
                cur_windows = list(np.array(frame_queue))

                #아래 np.array가 학습 + 프로세싱에 넘겨줄 큐에 들어가야함
                print(type(np.array(frame_queue)))

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

            # if 'writing' in results[0]: #이때가 우리가 찾는 결과가 나왔을때임

            multiQ=mp.Queue()
            info=mp.Queue()
            multiQ.put(prev_queue)
            info.put([w,h])
            storeP=mp.Process(target=store_video,args=(multiQ,info))
            storeP.start()
            storeP.join()
            #비워주기
            prev_queue=deque()
                
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

def store_video(q,info):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    info=info.get()
    w=info[0]
    h=info[1]
    out=cv2.VideoWriter('camera/images/hi.mp4',fourcc,20,(w,h))
    for i in q.get():
        out.write(i)
    out.release()
def webcam_main():
    global frame_queue, camera, frame, results, threshold, sample_length, \
        data, test_pipeline, model, device, average_size, label, \
        result_queue, drawing_fps, inference_fps
    # global streaming
    # global prev_maxlen,sub_maxlen, prev_queue,sub_ins 
    global prev_queue,w,h
    global process_inst
    process_inst=subprocess()
    w=''
    h=''

    # # 1초에 15장 20초에 300장 , predict발생시점으로 부터 40초전~20초전까지 저장

    prev_queue=deque()

    average_size = 1
    threshold = 0.01
    drawing_fps = 20
    inference_fps = 4
    device = torch.device("cpu")

    model = init_recognizer('camera/ai_models/mmaction2/configs/recognition/tsn/tsn_r50_video_inference_1x1x3_100e_kinetics400_rgb.py', 'camera/ai_models/mmaction2/checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth', device=device)
    camera = cv2.VideoCapture(0)
    if w=='' and h=='':
        w=round(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        h=round(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
        # sub_ins=subthread(sub_maxlen)
        pw = Thread(target=webcam_stream, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)

        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


class subprocess():
    def __init__(self):
        now=timezone.now
        self.video_name=now #변경X
        self.storage_time=now #변경O
        self.next = ''
        self.w=''
        self.h=''
        self.subqueue=deque()
        self.count=0
# if __name__ == '__main__':
#     webcam_main()

# class subthread():
#     def __init__(self,sub_len):
#         now=timezone.now
#         self.video_name=now #변경X
#         self.storage_time=now #변경O
#         self.subqueue=deque(maxlen=sub_len) 
#         self.count=0
#     def store_mem(self,sub_len):
#         while self.count<sub_len:
#             self.subqueue.append(prev_queue.popleft())
#             self.count+=1
#     def store_db(self,sub_len):
#         while self.subqueue:
#             now=timezone.now()
#             t=now.strftime('%y%m%d_%H-%M-%S')
#             # # cv2.imwrite('camera/record_video/record_img/{}.jpg'.format(t),frame)
#             # print(type(jpegbytes))
#             content = ContentFile(self.subqueue.popleft())
#             img_model=Image()
#             img_model.datetime=self.video_name
#             img_model.image.save('{}.jpg'.format(t),content,save=False)
#             img_model.save()
#             self.count=0


    