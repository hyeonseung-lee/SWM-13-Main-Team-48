from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from django.views.decorators import gzip
from camera.ai_models.default import *
from camera.ai_models.mediapipe import *
#미디어파이프
import mediapipe as mp
from os.path import abspath, join, dirname


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
 