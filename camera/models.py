from django.db import models
from users.models import *
from django.utils.translation import gettext_lazy as _
import os
from django.utils import timezone

# Create your models here.


class Video(models.Model):
    ymd = timezone.now().strftime("%Y%m%d")
    path = os.path.dirname(os.path.dirname(__file__))
    videopath = os.path.join(path, 'media', 'record_video', ymd)
    thumbpath=os.path.join(path,'media','record_img',ymd)

    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, null=True, blank=True, db_column='profile')
    camera=models.ForeignKey(Camera,on_delete=models.CASCADE, null=True, blank=True, db_column='camera')
    video = models.FilePathField(verbose_name=_("영상"), null=True, blank=True, path=videopath)
    type = models.CharField(verbose_name=_('상태'), max_length=100, null=True, blank=True)
    # datetime = models.DateTimeField(verbose_name=_("날짜"), auto_now_add=True)
    datetime = models.DateTimeField(verbose_name=_("날짜"))  # 해당 시간 입력해줄거기때문
    thumbnail= models.FilePathField(verbose_name=_("섬네일"), null=True, blank=True,path=thumbpath)
    #썸네일 추가

