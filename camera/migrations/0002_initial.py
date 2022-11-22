# Generated by Django 3.2 on 2022-11-23 01:30

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('camera', '0001_initial'),
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='video',
            name='camera',
            field=models.ForeignKey(blank=True, db_column='camera', null=True, on_delete=django.db.models.deletion.CASCADE, to='users.camera'),
        ),
        migrations.AddField(
            model_name='video',
            name='profile',
            field=models.ForeignKey(blank=True, db_column='profile', null=True, on_delete=django.db.models.deletion.CASCADE, to='users.profile'),
        ),
    ]
