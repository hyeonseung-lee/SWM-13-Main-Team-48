importScripts("https://www.gstatic.com/firebasejs/8.10.0/firebase-app.js");
importScripts("https://www.gstatic.com/firebasejs/8.10.0/firebase-messaging.js");

// firebase.initializeApp({
//     messagingSenderId: "1057826512183"
//   });
const firebaseConfig = {
  apiKey: "AIzaSyCxObFD0D39E7e3BrGwJPAf7aYhJaeZcMY",
  authDomain: "fcm-ai-cctv.firebaseapp.com",
  projectId: "fcm-ai-cctv",
  storageBucket: "fcm-ai-cctv.appspot.com",
  messagingSenderId: "1057826512183",
  appId: "1:1057826512183:web:d777f43a178163f20bceb2",
  measurementId: "G-X6NR2GX2TH"
};
firebase.initializeApp(firebaseConfig);
const messaging = firebase.messaging();

messaging.setBackgroundMessageHandler(payload => {
    const notification = JSON.parse(payload.data.notification);
    const notificationTitle = notification.title;
    const notificationOptions = {
      body: notification.body
    };
    //Show the notification :)
    return self.registration.showNotification(
      notificationTitle,
      notificationOptions
    );
  });
