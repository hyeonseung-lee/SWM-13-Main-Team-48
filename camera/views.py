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



# ------------- 아래가 주요 사용 됨 ---------------
def webcam_thread(request):
    return StreamingHttpResponse(webcam_thread_main(request),content_type="multipart/x-mixed-replace;boundary=frame")


def find(request):
   
    try:

        start_date=datetime.strptime(request.GET['start'],'%m/%d/%Y')
        end_date=datetime.strptime(request.GET['end'],'%m/%d/%Y')
        videos = Video.objects.filter(Q(datetime__lte=end_date)&Q(datetime__gte=start_date)&Q(profile=request.user.profile)).order_by('-datetime')
        # result=[]
        for i in videos:
            path='/'.join(i.video.split('/')[-4:])
            path=os.path.join('../../',path)
            i.video=path
            # result.append(path)
        return render(request, 'find.html',{"videos":videos})

    except:
        return render(request, 'find.html')

