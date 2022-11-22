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


urlpatterns = [
    path('admin/', admin.site.urls),
    path('camera/', include('camera.urls')),
    path('', main, name='dashboards'),
    path('list/', video_list, name='video_list'),
    path('users/', include('users.urls')),
    path('profile', profile, name='profile'),
    path('', include('pwa.urls')),

    path('token_save/',token_save,name="token_save"),
    path('token_delete/',token_delete,name="token_delete"),
    path('firebase-messaging-sw.js', firebase_messaging_sw,name='firebase_messaging_sw'),
    # firebase cloud message
    # path("firebase-messaging-sw.js",
    #      TemplateView.as_view(
    #          template_name="firebase-messaging-sw.js",
    #          content_type="application/javascript",
    #      ),
    #      name="firebase-messaging-sw"
    #      ),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
