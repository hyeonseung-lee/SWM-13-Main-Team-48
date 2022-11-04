from django.test import TestCase
from users.models import *
from camera.models import *
from django.utils import timezone

# Create your tests here.

class VideoStore(TestCase):
    def setUp(self)->None:
        super().setUp()

        self.사용자=User.objects.create()

    def test_(self):
        video=Video.objects.create(
        profile=self.사용자.profile,
        video='1667292909.mp4',
        type='fixing hair',
        datetime=timezone.now()
        )
        print(video.video)
        # res=self.client.post(
        #     path='/short-links',
        #     data={
        #         "url":"https://airbridge.io"
        #     },
        #     content_type="application/json"
        # )
        # self.assertEqual(res.status_code,201)

        # data=res.json()
        # re={
        #     "data": {
        #         "shortId": "baa",
        #         "url": "https://airbridge.io",
        #         "createdAt": "2022-10-31 03:56:01.541964"
        #     }
        # }
        # # print(data)
        # self.assertEqual(data['data']['shortId'],re['data']['shortId'])
        # self.assertEqual(data['data']['url'],re['data']['url'])