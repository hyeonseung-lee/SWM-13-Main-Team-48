from django.shortcuts import render
from django.contrib.auth.decorators import login_required


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
    return render(request, 'main.html', {"state": state})

@login_required
def video_list(request):
    dummy_videos = [
        {'url': "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
            'action_type': "실신", 'datetime': "2022-10-29 10:32"},
        {'url': "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
            'action_type': "장시간 배회", 'datetime': "2022-10-28 23:33"},
        {'url': "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4",
            'action_type': "기물파손", 'datetime': "2022-10-26 01:22"},
    ]
    return render(request, 'video_list.html', {"dummy_videos": dummy_videos})

@login_required
def profile(request):
    return render(request, 'profile.html')
