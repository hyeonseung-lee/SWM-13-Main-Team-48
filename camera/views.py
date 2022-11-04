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
from django.views.decorators.http import require_POST
from django.http import HttpResponse
from collections import deque
import threading
from django.utils import timezone
from django.conf import settings

from .models import test as test_model
import os

def camerapage(request):


    return render(request,'camera.html')
    
@gzip.gzip_page
def livecam(request):
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
    try:
        date=request.GET['date'].replace('-','')
        count=Video.objects.filter(video__contains=date).count()

        return render(request,'find.html',{'date':date,'count':count})
    except:
        return render(request,'find.html')

def show(request,date):
    videos=Video.objects.all()
    result=[]
    for i in videos:
        path='/'.join(i.video.split('/')[-4:])
        # print(path)
        path=os.path.join('../../',path)
        result.append(path)
    return render(request,'show.html',{'date':date,"videos":result})

def videostream(request,date):
    return StreamingHttpResponse(read_and_encode(date),content_type="multipart/x-mixed-replace;boundary=frame")

def read_and_encode(date):
    # img=cv2.imdecode(image_byte_path,1)
    # print(img)
    all=Video.objects.filter(image__contains=date)
    for i in all:
        image=cv2.imread(i.image.path,cv2.IMREAD_COLOR)
        _, jpg = cv2.imencode('.jpg', image)
        frame = jpg.tobytes()
   
        yield(b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')

def test(request):
    if request.POST:
        a=test_model.objects.create(
            test=request.FILES['file']
        )
        print(type(a.test))
    return render(request,'test.html')

def test1(request):
    # if request.POST:
    #     a=Video.objects.create(
    #         profile=request.user.profile,
    #         video='media/record_video/20221104/221104_20-11-43.mp4',
    #         datetime=timezone.now()
    #     )
    #     # print(a.video)
    # video='media/record_video/20221104/221104_20-11-43.mp4'
    # t=video.split('/')

    # save_path = os.path.dirname(os.path.dirname(__file__))
    # save_path = os.path.join(save_path[:-7], 'media','record_video') 

    # print(t)
    videos=Video.objects.all()
    result=[]
    for i in videos:
        path='/'.join(i.video.split('/')[-4:])
        # print(path)
        path=os.path.join('../../',path)
        result.append(path)
    #     print('--------------------')
    #     filename=os.path.basename(i.video)
    #     path=os.path.join('../../','media','')
    #     print(os.path.exists(i.video))
        

    return render(request,'test.html',{"videos":videos})