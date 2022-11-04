from django.urls import path, include
from camera.views import *

app_name = "camera"

urlpatterns = [
	path('camerapage', camerapage, name='camerapage'),
	path('livecam', livecam, name='livecam'),
	path('mediapipe', mediapipe, name='mediapipe'),
	path('mediapipe_with_yolo', mediapipe_with_yolo, name='mediapipe_with_yolo'),
	path('enhancement_yolo', enhancement_yolo, name='enhancement_yolo'),
	
	# 일반 영상
	path('mmaction2', mmaction2, name='mmaction2'),
	# webcam demo
	path('webcam_demo', webcam_demo, name='webcam_demo'),

	path('webcam_thread', webcam_thread, name='webcam_thread'),

	path('find',find,name='find'),
	path('show/<str:date>',show,name='show'),

	path('test',test,name='test'),
	path('test1',test1,name='test1'),

    
]
