from django.shortcuts import render


def main(request):
    state = False
    return render(request, 'main.html', )


def video_list(request):
    dummy_videos = [
        {'url': "/######", 'action_type': "실신", 'datetime': "2022-10-29 10:32"},
        {'url': "/######", 'action_type': "장시간 배회", 'datetime': "2022-10-28 23:33"},
        {'url': "/######", 'action_type': "기물파손", 'datetime': "2022-10-26 01:22"},
    ]
    return render(request, 'video_list.html', {"dummy_videos": dummy_videos})
