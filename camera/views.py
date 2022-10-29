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
from django.db.models import Q
from django.views.decorators.http import require_POST
from django.http import HttpResponse

from os.path import abspath, join, dirname


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
    return StreamingHttpResponse(mmaction_stream(), content_type="multipart/x-mixed-replace;boundary=frame")


def webcam_demo(request):
    # return StreamingHttpResponse([] if webcam_main() is None else webcam_main(),content_type="multipart/x-mixed-replace;boundary=frame")
    return StreamingHttpResponse(webcam_main(), content_type="multipart/x-mixed-replace;boundary=frame")


def find(request):
    dummy_videos = [
        {'url': "/######", 'action_type': "실신", 'datetime': "2022-10-29 10:32"},
        {'url': "/######", 'action_type': "장시간 배회", 'datetime': "2022-10-28 23:33"},
        {'url': "/######", 'action_type': "기물파손", 'datetime': "2022-10-26 01:22"},
    ]
    try:
        start = request.GET['start'].split("/")
        start_date = start[2] + start[0] + start[1]
        end = request.GET['end'].replace('/', '')
        end_date = end[2] + end[0] + end[1]
        date = start + "-" + end
        count = Image.objects.filter(
            Q(image__lte=end_date) & Q(image__gte=start_date)).count()

        return render(request, 'find.html', {'date': date, 'count': count, 'dummy_videos': dummy_videos}, )
    except:
        return render(request, 'find.html')


def show(request, date):
    return render(request, 'show.html', {'date': date})


def videostream(request, date):
    return StreamingHttpResponse(read_and_encode(date), content_type="multipart/x-mixed-replace;boundary=frame")


def read_and_encode(date):
    # img=cv2.imdecode(image_byte_path,1)
    # print(img)
    all = Image.objects.filter(image__contains=date)
    for i in all:
        image = cv2.imread(i.image.path, cv2.IMREAD_COLOR)
        _, jpg = cv2.imencode('.jpg', image)
        frame = jpg.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n\r\n')

# @require_POST
# def po_check(request):

#     poname=json.loads(request.body)

#     poname=poname['po']
#     # print(poname)
#     polist={}

#     if poname == '': # 이름 쳤다가 다지워서 빈게 contains인 경우 제외
#         list=[]
#     else:
#         list=Company.objects.filter(company_name__contains=poname)

#         if list.count()!=1 or (list.count()==1 and list.first().company_name!=poname):
#             for i in list:
#                 polist[i.company_code]=i.company_name

#     # print(polist)
#     return HttpResponse(json.dumps(polist),content_type="application/json")
