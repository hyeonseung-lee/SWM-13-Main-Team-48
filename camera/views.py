from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from django.views.decorators import gzip
from camera.ai_models.check import *
# Create your views here.

def camerapage(request):
    return render(request,'camera.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@gzip.gzip_page
def livecam(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")
        pass
