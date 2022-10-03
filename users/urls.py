from .views import *
from django.urls import path
from django.urls.conf import include
from rest_framework_simplejwt.views import TokenBlacklistView,TokenRefreshView

app_name="users"

urlpatterns = [
    path('kakao_login',kakao_login,name="kakao_login"),
    path('kakao/login/callback/',get_token,name="get_token"),
    
    path('profile_update',profile_update,name="profile_update"),
    path('kakao_logout',kakao_logout,name="kakao_logout"),
    
    # django 로그아웃
    path('logout',TokenBlacklistView.as_view(),name="logout"),
    # django 로그인 용(superuser사용)
    path('login',MyTokenObtainPairView.as_view(),name="login"),
    
    #access토큰 재발급, {"refresh": 토큰} 꼴로 post
    path('token-refresh', TokenRefreshView.as_view(), name='token_refresh'),
]