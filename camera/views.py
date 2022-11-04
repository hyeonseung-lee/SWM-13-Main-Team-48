from ast import Index
from multiprocessing import dummy
from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from django.views.decorators import gzip
from camera.ai_models.default import *
from camera.ai_models.mediapipe import *
from camera.ai_models.mediapipe_with_yolo import *
from camera.ai_models.enhancement_yolo import *
from camera.ai_models.mmaction import *
from camera.ai_models.webcam import *
from camera.ai_models.webcam_thread import *
from .models import *
from django.db.models import Q
from django.http import HttpResponse
from collections import deque
from django.utils import timezone
from django.conf import settings
from datetime import datetime
import os

def camerapage(request):
    return render(request, 'camera.html')


@gzip.gzip_page
def livecam(request):
    # template으로 넘길 수는 없는지
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass


@gzip.gzip_page
def mediapipe(request):
    return StreamingHttpResponse(mediapipe_stream(), content_type="multipart/x-mixed-replace;boundary=frame")


@gzip.gzip_page
def mediapipe_with_yolo(request):
    return StreamingHttpResponse(mediapipe_with_yolo_stream(), content_type="multipart/x-mixed-replace;boundary=frame")


@gzip.gzip_page
def enhancement_yolo(request):
    return StreamingHttpResponse(enhancement_yolo_stream(), content_type="multipart/x-mixed-replace;boundary=frame")


def mmaction2(request):
   
    return StreamingHttpResponse(mmaction_stream(),content_type="multipart/x-mixed-replace;boundary=frame")

def webcam_demo(request):
        
    return StreamingHttpResponse(webcam_main(),content_type="multipart/x-mixed-replace;boundary=frame")

def webcam_thread(request):
        
    return StreamingHttpResponse(webcam_thread_main(request),content_type="multipart/x-mixed-replace;boundary=frame")


def find(request):
    # dummy_videos = [
    #     {'url': "/######", 'action_type': "실신", 'datetime': "2022-10-29 10:32"},
    #     {'url': "/######", 'action_type': "장시간 배회", 'datetime': "2022-10-28 23:33"},
    #     {'url': "/######", 'action_type': "기물파손", 'datetime': "2022-10-26 01:22"},
    # ]
    # start = request.GET['start'].split("/")
    # start_date = start[2] + start[0] + start[1]
    # end = request.GET['end'].replace('/', '')
    # end_date = end[2] + end[0] + end[1]
    # date = start + "-" + end

    # print(start)
    # print(start_date)
    # print(end)
    # print(end_date)
    # print(date)

    # count = Video.objects.filter(
    #     Q(datetime__lte=end_date) & Q(datetime__gte=start_date)).count()
    # print(count)
    # return render(request, 'find.html', {'date': date, 'count': count} )
    # print('hi')
    try:

        start_date=datetime.strptime(request.GET['start'],'%m/%d/%Y')
        end_date=datetime.strptime(request.GET['end'],'%m/%d/%Y')
        videos = Video.objects.filter(Q(datetime__lte=end_date)&Q(datetime__gte=start_date)).order_by('-datetime')
        # result=[]
        for i in videos:
            path='/'.join(i.video.split('/')[-4:])
            path=os.path.join('../../',path)
            i.video=path
            # result.append(path)
        return render(request, 'find.html',{"videos":videos})

    except:
        return render(request, 'find.html')



def show(request,date):
    videos=Video.objects.all()
    result=[]
    for i in videos:
        path='/'.join(i.video.split('/')[-4:])
        path=os.path.join('../../',path)
        result.append(path)
    return render(request,'show.html',{'date':date,"videos":result})


# def test(request):
#     if request.POST:
#         a=test_model.objects.create(
#             test=request.FILES['file']
#         )
#         print(type(a.test))
#     return render(request,'test.html')

# def test1(request):
#     # if request.POST:
#     #     a=Video.objects.create(
#     #         profile=request.user.profile,
#     #         video='media/record_video/20221104/221104_20-11-43.mp4',
#     #         datetime=timezone.now()
#     #     )
#     #     # print(a.video)
#     # video='media/record_video/20221104/221104_20-11-43.mp4'
#     # t=video.split('/')

#     # save_path = os.path.dirname(os.path.dirname(__file__))
#     # save_path = os.path.join(save_path[:-7], 'media','record_video') 

#     # print(t)
#     videos=Video.objects.all()
#     result=[]
#     for i in videos:
#         path='/'.join(i.video.split('/')[-4:])
#         # print(path)
#         path=os.path.join('../../',path)
#         result.append(path)
#     #     print('--------------------')
#     #     filename=os.path.basename(i.video)
#     #     path=os.path.join('../../','media','')
#     #     print(os.path.exists(i.video))
        

#     return render(request,'test.html',{"videos":videos})
