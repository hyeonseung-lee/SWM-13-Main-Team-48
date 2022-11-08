from firebase_admin import messaging

def send_to_firebase_cloud_messaging_User(fcm_token,user_info,behavior):
    # This registration token comes from the client FCM SDKs.
    registration_token = fcm_token

    # See documentation on defining a message payload.
    message = messaging.Message(
        notification=messaging.Notification(
            title='이상행동 감지로 인한 영상 저장 안내',
            body='{0}의 가게에서 {1}와 같은 이상행동이 발생했습니다'.format(user_info,behavior),
            
        ),
        token=registration_token,
    )

    response = messaging.send(message)
    # Response is a message ID string.
    print('Successfully sent message:', response)
