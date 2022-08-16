from django.urls import path, include
from camera.views import *

app_name="camera"

urlpatterns = [
	path('camerapage', camerapage, name='camerapage'),
	path('livecam', livecam, name='livecam'),
	path('mediapipe', mediapipe, name='mediapipe'),
	path('mediapipe_with_yolo', mediapipe_with_yolo, name='mediapipe_with_yolo'),
	path('enhancement_yolo', enhancement_yolo, name='enhancement_yolo'),

    
]
