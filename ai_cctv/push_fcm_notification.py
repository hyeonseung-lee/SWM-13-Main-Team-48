from firebase_admin import messaging


def send_to_firebase_cloud_messaging():
    # This registration token comes from the client FCM SDKs.
    registration_token = 'c2wy4Ij2fQ0:APA91bHaeDjLbwrEVSh_EZo-kmrYntB9aSiVcX2NQ53drfSdu_x-UKs5sujOojfHtrOrNlYHbhUBn5v6dsbd3BmGcumUoHsszNL75P506y5RWfCmoaYPDpF3GnS0kVFwxSS5x_eWZVAM'

    # See documentation on defining a message payload.
    message = messaging.Message(
        notification=messaging.Notification(
            title='안녕하세요 타이틀 입니다',
            body='안녕하세요 메세지 입니다',
        ),
        token=registration_token,
    )
    response = messaging.send(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)
    # try:
    #     response = messaging.send(message)
    #     # Response is a message ID string.
    #     print('Successfully sent message:', response)
    # except Exception as e:
    #     print('예외가 발생했습니다.', e)
