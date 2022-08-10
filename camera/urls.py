from django.urls import path, include
from camera.views import *

app_name="camera"

urlpatterns = [
	path('camerapage', camerapage, name='camerapage'),
	path('livecam', livecam, name='livecam'),
	path('mediapipe', mediapipe, name='mediapipe'),

    
]
