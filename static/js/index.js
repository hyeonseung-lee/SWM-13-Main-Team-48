// Initialize Firebase
// Firebase Console --> Settings --> General
// --> Register App --> Copy firebaseConfig
// For Firebase JS SDK v7.20.0 and later, measurementId is optional

const firebaseConfig = {
  apiKey: "AIzaSyCxObFD0D39E7e3BrGwJPAf7aYhJaeZcMY",
  authDomain: "fcm-ai-cctv.firebaseapp.com",
  projectId: "fcm-ai-cctv",
  storageBucket: "fcm-ai-cctv.appspot.com",
  messagingSenderId: "1057826512183",
  appId: "1:1057826512183:web:d777f43a178163f20bceb2",
  measurementId: "G-X6NR2GX2TH",
};

firebase.initializeApp(firebaseConfig);

// Firebase Messaging Service
const messaging = firebase.messaging();

function sendTokenToServer(currentToken) {
  var $crf_token = $('[name="csrfmiddlewaretoken"]').attr("value");
  if (!isTokenSentToServer()) {
    // The API Endpoint will be explained at step 8
    $.ajax({
      url: "/api/devices/",
      method: "POST",
      async: false,
      data: {
        registration_id: currentToken,
        type: "web",
      },
      headers: {
        "X-CSRFTOKEN": $crf_token,
      },
      success: function (data) {
        console.log(data);
        setTokenSentToServer(true);
      },
      error: function (err) {
        console.log(err);
        setTokenSentToServer(false);
      },
    });
  } else {
    console.log(
      "Token already sent to server so won't send it again " +
        "unless it changes"
    );
  }
}

function isTokenSentToServer() {
  return window.localStorage.getItem("sentToServer") === "1";
}

function setTokenSentToServer(sent) {
  if (sent) {
    window.localStorage.setItem("sentToServer", "1");
  } else {
    window.localStorage.setItem("sentToServer", "0");
  }
}

function requestPermission() {
  messaging
    .requestPermission()
    .then(function () {
      console.log("Has permission!");
      resetUI();
    })
    .catch(function (err) {
      console.log("Unable to get permission to notify.", err);
    });
}

function resetUI() {
  console.log("In reset ui");
  messaging
    .getToken()
    .then(function (currentToken) {
      console.log(currentToken);
      if (currentToken) {
        sendTokenToServer(currentToken);
      } else {
        setTokenSentToServer(false);
      }
    })
    .catch(function (err) {
      console.log(err);
      setTokenSentToServer(false);
    });
}

messaging.onTokenRefresh(function () {
  messaging
    .getToken()
    .then(function (refreshedToken) {
      console.log("Token refreshed.");
      // Indicate that the new Instance ID token has not yet been sent to the
      // app server.
      setTokenSentToServer(false);
      // Send Instance ID token to app server.
      sendTokenToServer(refreshedToken);
      resetUI();
    })
    .catch(function (err) {
      console.log("Unable to retrieve refreshed token ", err);
    });
});

messaging.onMessage(function (payload) {
  payload = payload.data;
  // Create notification manually when user is focused on the tab
  const notificationTitle = payload.title;
  const notificationOptions = {
    body: payload.body,
    icon: payload.icon_url,
  };

  if (!("Notification" in window)) {
    console.log("This browser does not support system notifications");
  }
  // Let's check whether notification permissions have already been granted
  else if (Notification.permission === "granted") {
    // If it's okay let's create a notification
    var notification = new Notification(notificationTitle, notificationOptions);
    notification.onclick = function (event) {
      event.preventDefault(); // prevent the browser from focusing the Notification's tab
      window.open(payload.url, "_blank");
      notification.close();
    };
  }
});

requestPermission();
