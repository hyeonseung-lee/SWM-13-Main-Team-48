# Generated by Django 3.2 on 2022-11-13 03:42

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('video', models.FilePathField(blank=True, null=True, path='/Users/tae.spider_oo/Desktop/real/ai_cctv/media/record_video/20221113', verbose_name='영상')),
                ('type', models.CharField(blank=True, max_length=100, null=True, verbose_name='상태')),
                ('datetime', models.DateTimeField(verbose_name='날짜')),
                ('thumbnail', models.FilePathField(blank=True, null=True, path='/Users/tae.spider_oo/Desktop/real/ai_cctv/media/record_img/20221113', verbose_name='섬네일')),
            ],
        ),
    ]
