from django.contrib.auth.decorators import login_required
from fcm_django.models import FCMDevice
from firebase_admin.messaging import Message
from .push_fcm_notification import send_to_firebase_cloud_messaging
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .push_fcm_notification import send_to_firebase_cloud_messaging
from django.contrib import messages
from firebase_admin.messaging import Message
from fcm_django.models import FCMDevice
from time import sleep
from users.models import *
from camera.models import *
from django.db.models import Q, Count
from django.utils import timezone
import datetime


def main(request):
    try:
        # 알람 설정하기
        alarm_state = False

        today = timezone.now().date()
        tomorrow = today+datetime.timedelta(days=1)
        main_store = request.user.profile.main_store

        main_store_cameras = Camera.objects.filter(store=main_store)
        print(main_store)
        if main_store:
            # 전체 매장 - 드롭다운용
            stores = Store.objects.filter(
                owner=request.user).exclude(id=main_store.id)
            print(stores)

            # daily_cumulative_detection=0
            daily_cumulative_detection = 0
            # value['swooner']=0
            # value['vandalism']=0
            # value['violence']=0

            for camera in main_store_cameras:

                # 한 카메라에 대한 하루 비디오 정보
                videos = Video.objects.filter(
                    Q(datetime__gte=today) & Q(datetime__lte=tomorrow) & Q(camera=camera))

                daily_cumulative_detection += videos.count()

            #     video_type=videos.values('type').annotate(type_count=Count('type'))
            #     for i in video_type:

            #         # print(i.type_count)
            #         # print(i.type)
            # from DB
            # from Yolo
            # print(main_store)
            context = {
                "main_store": main_store,
                "stores": stores,
                "daily_cumulative_detection": daily_cumulative_detection
            }

        # sleep(20)
        # send_to_firebase_cloud_messaging()
        """ test git README """
        # message_obj = Message(
        #     data={
        #         "Nick": "Mario",
        #         "body": "great match!",
        #         "Room": "PortugalVSDenmark"
        #     },
        # )

        # You can still use .filter() or any methods that return QuerySet (from the chain)
        # device = FCMDevice.objects.all().first()
        # device.send_message(message_obj)

        return render(request, 'main.html', {"context": context, "alarm_state": alarm_state})
    except:
        return render(request, 'main.html')


@login_required(login_url='dashboards')
def video_list(request):
    try:
        today = timezone.now().date()
        tomorrow = today+datetime.timedelta(days=1)
        main_store = request.user.profile.main_store
        main_store_cameras = Camera.objects.filter(store=main_store)
        if main_store:
            # 전체 매장 - 드롭다운용
            stores = Store.objects.filter(
                owner=request.user).exclude(id=main_store.id)

            video_l = []
            for camera in main_store_cameras:

                if request.GET:
                    start_date = datetime.datetime.strptime(
                        request.GET['start'], '%m/%d/%Y')
                    end_date = datetime.datetime.strptime(
                        request.GET['end'], '%m/%d/%Y')
                    videos = Video.objects.filter(Q(datetime__lte=end_date) & Q(
                        datetime__gte=start_date) & Q(camera=camera)).order_by('-datetime')
                else:
                    # 한 카메라에 대한 하루 비디오 정보
                    videos = Video.objects.filter(
                        Q(datetime__gte=today) & Q(datetime__lte=tomorrow) & Q(camera=camera)).order_by('-datetime')
                for i in videos:
                    video_path = '/'.join(i.video.split('/')[-4:])
                    video_path = os.path.join('../../', video_path)
                    i.video = video_path

                    image_path = '/'.join(i.thumbnail.split('/')[-4:])
                    image_path = os.path.join('../../', image_path)
                    i.thumbnail = image_path
                video_l.append(videos)
            print(video_l)
            context = {
                "stores": stores,
                "video_list": video_l,
                "main_store": main_store
            }
            return render(request, 'video_list.html', {"context": context})
        else:
            print('메인 매장이없음')
            messages.warning(request, "메인 매장이 없습니다. 설정하세요")
            return redirect('dashboards')
    except:
        return render(request, 'video_list.html')
    # test_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"

    # action_type = ["영업방해 의심행위", "시설물 파손 의심행위"]

    # swoon_description = "실신, 매장 내 드러누움 등 의심"
    # vandalism_description = "매장 내 시설물 파손 등 시설물 이상 의심"
    # dummy_videos = [
    #     {'url': test_url,
    #         'action_type': action_type[0], "description": swoon_description, 'datetime': "2022-10-29 10:32", "store":"store"},
    #     {'url': test_url,
    #         'action_type': action_type[0], "description": swoon_description, 'datetime': "2022-10-28 23:33", "store":"store"},
    #     {'url': test_url,
    #         'action_type': action_type[1], "description": vandalism_description, 'datetime': "2022-10-26 01:22", "store":"store"},
    # ]


@login_required(login_url='dashboards')
def profile(request):
    return render(request, 'profile.html')


def test(request):
    return render(request, 'test.html')
