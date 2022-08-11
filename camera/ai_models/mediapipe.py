import mediapipe as mp
from os.path import abspath, join, dirname
import cv2

def mediapipe_stream():

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # For webcam input:
    cap = cv2.VideoCapture('rtsp://admin:somateam23@aicctv.iptime.org:554/profile2/media.smp')
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            backgroundimage = abspath( join(dirname(__file__), "../images/white.jpg") )

            background = cv2.imread(backgroundimage, cv2.IMREAD_COLOR)
            background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
            background = cv2.resize(background, (image.shape[1], image.shape[0]))
            
            mp_drawing.draw_landmarks(
                background,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            _, jpeg = cv2.imencode('.jpg', background)
            frame=jpeg.tobytes()

            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
            # # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Pose', cv2.flip(background, 1))
            # cv2.imwrite(f'save_images/test.png', background)
            # if cv2.waitKey(5) & 0xFF == 27: # esc누르면 break
            #     break
        cap.release() #오픈한 캡쳐 객체 닫기
