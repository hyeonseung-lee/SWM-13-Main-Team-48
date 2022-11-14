"""ai_cctv URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import *

from django.views.generic import TemplateView  # fcm
from fcm_django.api.rest_framework import FCMDeviceAuthorizedViewSet  # fcm
from rest_framework.routers import DefaultRouter  # fcm

""" for firebase cloud message """
router = DefaultRouter()
router.register('devices', FCMDeviceAuthorizedViewSet)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('camera/', include('camera.urls')),
    path('', main, name='dashboards'),
    path('list/', video_list, name='video_list'),
    path('users/', include('users.urls')),
    path('profile', profile, name='profile'),
    path('', include('pwa.urls')),
    path('', include('pwa_webpush.urls')),

    path('test', test),

    # firebase cloud message
    path("firebase-messaging-sw.js",
         TemplateView.as_view(
             template_name="firebase-messaging-sw.js",
             content_type="application/javascript",
         ),
         name="firebase-messaging-sw.js"
         ),
    path('api/', include(router.urls)),
    # path('accounts/', include('allauth.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
