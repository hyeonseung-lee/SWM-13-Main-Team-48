from tabnanny import verbose
from django.db import models
from users.models import *
from django.utils.translation import gettext_lazy as _

# Create your models here.

class Image(models.Model):
    profile=models.ForeignKey(Profile, on_delete=models.CASCADE, null=True, blank=True, db_column='profile') 
    # video_clip=models.CharField(verbose_name=_("비디오 클립"), max_length=200,null=True,blank=True)
    image=models.FileField(verbose_name=_("이미지"),upload_to='record_video/%Y%m%d/',null=True,blank=True) 
    # level = models.CharField(verbose_name=_('상태'),max_length=100)
    datetime = models.DateTimeField(verbose_name=_("날짜"), auto_now=False)
    group = models.CharField(verbose_name=_('같은클립'), max_length=100 ,null=True,blank=True)

class Camera(models.Model):
    profile=models.ForeignKey(Profile, on_delete=models.CASCADE, null=True, blank=True, db_column='profile') 
    rtsp_url=models.CharField(verbose_name=_("rtsp url"), max_length=100,null=True, blank=True) 
