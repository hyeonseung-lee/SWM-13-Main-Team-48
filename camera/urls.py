from django.urls import path, include
from camera.views import *

app_name = "camera"

urlpatterns = [
    # path('camerapage', camerapage, name='camerapage'),
    # path('mediapipe', mediapipe, name='mediapipe'),
    # path('mediapipe_with_yolo', mediapipe_with_yolo, name='mediapipe_with_yolo'),
    # path('enhancement_yolo', enhancement_yolo, name='enhancement_yolo'),
    path('default', default, name='default'),
    # path('webcam_demo', webcam_demo, name='webcam_demo'),

    # path('livecam', livecam, name='livecam'),

    path('webcam_thread', webcam_thread, name='webcam_thread'),

    # path('cam_yolo7_multiprocessing', cam_yolo7_multiprocessing, name='cam_yolo7_multiprocessing'),
    path('cam_yolo5_multiprocessing', cam_yolo5_multiprocessing, name='cam_yolo5_multiprocessing'),

    path('find', find, name='find'),
    path('livepage', livepage, name='livepage'),

    path('test', test, name='test')
]
