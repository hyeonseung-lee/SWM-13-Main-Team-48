from django.contrib.auth.decorators import login_required
from fcm_django.models import FCMDevice
from firebase_admin.messaging import Message
from .push_fcm_notification import send_to_firebase_cloud_messaging
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .push_fcm_notification import send_to_firebase_cloud_messaging

from firebase_admin.messaging import Message
from fcm_django.models import FCMDevice


def main(request):
    # from DB
    # from Yolo
    visitor = 10    # 방문자 수
    wandor = 2      # 배회
    # from MMAction2
    swooner = 4     # 실신
    vandalism = 2   # 기물파손
    violence = 3    # 폭행

    # set to front-end
    # visitor = visitor
    obstructions = wandor + swooner + violence  # 영업방해행위 (배회 + 실신 + 폭행)
    # vandalism = vandalism
    state = {"visitor": visitor,
             "obstructions": obstructions, "vandalism": vandalism}
    # send_to_firebase_cloud_messaging()

    """ test git README """
    message_obj = Message(
        data={
            "Nick": "Mario",
            "body": "great match!",
            "Room": "PortugalVSDenmark"
        },
    )

    # You can still use .filter() or any methods that return QuerySet (from the chain)
    device = FCMDevice.objects.all().first()

    # send_message parameters include: message, dry_run, app
    # while True:
    # device.send_message(message_obj)
    print(device)
    # Boom!

    return render(request, 'main.html', {"state": state})


@login_required(login_url='dashboards')
def video_list(request):
    test_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4"

    action_type = ["영업방해 의심행위", "시설물 파손 의심행위"]

    swoon_description = "실신, 매장 내 드러누움 등 의심"
    vandalism_description = "매장 내 시설물 파손 등 시설물 이상 의심"
    dummy_videos = [
        {'url': test_url,
            'action_type': action_type[0], "description": swoon_description, 'datetime': "2022-10-29 10:32", "store":"STORE_NAME"},
        {'url': test_url,
            'action_type': action_type[0], "description": swoon_description, 'datetime': "2022-10-28 23:33", "store":"STORE_NAME"},
        {'url': test_url,
            'action_type': action_type[1], "description": vandalism_description, 'datetime': "2022-10-26 01:22", "store":"STORE_NAME"},
    ]
    return render(request, 'video_list.html', {"dummy_videos": dummy_videos})


@login_required(login_url='dashboards')
def profile(request):
    return render(request, 'profile.html')


def test(request):
    return render(request, 'test.html')
