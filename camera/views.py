from ast import Index
from multiprocessing import dummy
from django.shortcuts import render, redirect
from django.http.response import StreamingHttpResponse
from django.views.decorators import gzip
from camera.ai_models.default import *
# from camera.ai_models.mediapipe import *
# from camera.ai_models.mediapipe_with_yolo import *
# from camera.ai_models.enhancement_yolo import *
from camera.ai_models.default2 import *
# from camera.ai_models.webcam import *
from camera.ai_models.webcam_thread import *
from camera.ai_models.cam_with_yolov7_in_multiprocessing import *
from camera.ai_models.cam_with_yolov5_in_multiprocessing import *
from .models import *
from django.db.models import Q
from django.http import HttpResponse
from collections import deque
from django.utils import timezone
from datetime import datetime
import os
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import torch
import sys

# cap=cv2.VideoCapture(0)
# @login_required(login_url='dashboards')
# def camerapage(request):
#     return render(request, 'camera.html')

@gzip.gzip_page
@login_required(login_url='dashboards')
def livecam(request):
    # template으로 넘길 수는 없는지
    try:
        cam = VideoCamera()
        stream = StreamingHttpResponse(
            gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
        print(stream)
        print('hi')
        return stream
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass

def test(request):
    print('여기')
    print(torch.__version__)
    sys.path.insert(0,'./ai_models')
    
    a=torch.load('ai_modelsyolov7.pt')
    print(a)
    print('되나')

    return redirect('dashboards')

@login_required(login_url='dashboards')
def default(request):
    # global cap
    cap = cv2.VideoCapture()
    cap.open(0)

    # src=request.user.profile.main_store.default_cam
    # cap=cap(src)
    return StreamingHttpResponse(default_streaming(cap), content_type="multipart/x-mixed-replace;boundary=frame")

# @gzip.gzip_page
# @login_required(login_url='dashboards')
# def mediapipe(request):
#     return StreamingHttpResponse(mediapipe_stream(), content_type="multipart/x-mixed-replace;boundary=frame")

# @gzip.gzip_page
# @login_required(login_url='dashboards')
# def mediapipe_with_yolo(request):
#     return StreamingHttpResponse(mediapipe_with_yolo_stream(), content_type="multipart/x-mixed-replace;boundary=frame")

# @gzip.gzip_page
# @login_required(login_url='dashboards')
# def enhancement_yolo(request):
#     return StreamingHttpResponse(enhancement_yolo_stream(), content_type="multipart/x-mixed-replace;boundary=frame")


# @login_required(login_url='dashboards')
# def webcam_demo(request):
#     return StreamingHttpResponse(webcam_main(), content_type="multipart/x-mixed-replace;boundary=frame")


# ------------- 아래가 주요 사용 됨 ---------------
@login_required(login_url='dashboards')
def webcam_thread(request):
    main_store = request.user.profile.main_store
    if main_store is None:
        print('메인 매장이없음')
        messages.warning(request, "메인 매장이 없습니다. 설정하세요")
        return redirect('dashboards')
    default_camera = Camera.objects.filter(Q(store=main_store) &
                                           Q(main_cam=True))
    if default_camera.exists():
        default_camera = default_camera.first()
        print(default_camera)
        return StreamingHttpResponse(webcam_thread_main(request, default_camera), content_type="multipart/x-mixed-replace;boundary=frame")
    else:
        messages.warning(request, "디폴트 카메라가 없습니다. 설정하세요")
        return redirect('dashboards')


# @login_required(login_url='dashboards')
# def cam_yolo7_multiprocessing(request):
#     return StreamingHttpResponse(multiprocessing_main(), content_type="multipart/x-mixed-replace;boundary=frame")

@login_required(login_url='dashboards')
def cam_yolo5_multiprocessing(request):
    main_store = request.user.profile.main_store
    if main_store is None:
        print('메인 매장이없음')
        messages.warning(request, "메인 매장이 없습니다. 설정하세요")
        return redirect('dashboards')
    default_camera = Camera.objects.filter(Q(store=main_store) &
                                           Q(main_cam=True))
    if default_camera.exists():
        default_camera = default_camera.first()
        print(default_camera)
        return StreamingHttpResponse(multiprocessing_main(request,default_camera) , content_type="multipart/x-mixed-replace;boundary=frame")
    else:
        messages.warning(request, "디폴트 카메라가 없습니다. 설정하세요")
        return redirect('dashboards')



@login_required(login_url='dashboards')
def find(request):

    try:
        main_store = request.user.profile.main_store
        if main_store is None:
            print('메인 매장이없음')
            messages.warning(request, "메인 매장이 없습니다. 설정하세요")
            return redirect('dashboards')
        default_camera = Camera.objects.filter(store=main_store, main_cam=True)
        if default_camera.exists():
            default_camera = default_camera.first()

            start_date = datetime.strptime(request.GET['start'], '%m/%d/%Y')
            end_date = datetime.strptime(request.GET['end'], '%m/%d/%Y')
            videos = Video.objects.filter(Q(datetime__lte=end_date) & Q(
                datetime__gte=start_date) & Q(camera=default_camera)).order_by('-datetime')

            # result=[]
            for i in videos:
                path = '/'.join(i.video.split('/')[-4:])
                path = os.path.join('../../', path)
                i.video = path
                # result.append(path)
            return render(request, 'find.html', {"videos": videos})
        else:
            messages.warning(request, "디폴트 카메라가 없습니다. 설정하세요")
            return redirect('dashboards')
    except:
        return render(request, 'find.html')


@login_required(login_url='dashboards')
def livepage(request):
    return render(request, 'livepage.html')
