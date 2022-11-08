from .views import *
from django.urls import path

app_name="users"

urlpatterns = [
    path('kakao_login',kakao_login,name="kakao_login"),
    path('kakao/login/callback/',get_token,name="get_token"),
    
    # path('profile_update',profile_update,name="profile_update"),
    path('kakao_logout',kakao_logout,name="kakao_logout"),
    
    # django 로그아웃
    path('logout',service_logout,name="logout"),
    path('kakao/logout/callback/',go_main,name="go_main"),
  
]