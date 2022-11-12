import cv2
import threading


class VideoCamera(object):
    def __init__(self):
        # self.video = cv2.VideoCapture('rtsp://admin:somateam23@172.16.101.184:554/profile2/media.smp')
        # self.video = cv2.VideoCapture('rtsp://admin:somateam23@aicctv.iptime.org:554/profile2/media.smp')
        self.video = cv2.VideoCapture(0)
        # self.video = cv2.VideoCapture('rtsp://admin:somateam23@172.16.100.251:554/profile2/media.smp')


        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
