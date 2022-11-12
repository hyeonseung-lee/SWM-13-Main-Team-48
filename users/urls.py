from .views import *
from django.urls import path

app_name = "users"

urlpatterns = [
    path('kakao_login', kakao_login, name="kakao_login"),
    path('kakao/login/callback/', get_token, name="get_token"),

    # path('profile_update',profile_update,name="profile_update"),
    path('kakao_logout', kakao_logout, name="kakao_logout"),

    # django 로그아웃
    path('logout', service_logout, name="logout"),
    path('kakao/logout/callback/', go_main, name="go_main"),
    # path('profile', profile, name='profile'),
    path('logout', service_logout, name="logout"),
    path('kakao/logout/callback/', go_main, name="go_main"),

    path('profile_update_page', profile_update_page, name="profile_update_page"),
    path('profile_update', profile_update, name="profile_update"),

    path('create_store_page', create_store_page, name="create_store_page"),
    path('create_store', create_store, name="create_store"),
    path('update_store_page/<str:store_id>',
         update_store_page, name="update_store_page"),
    path('update_store/<str:store_id>', update_store, name="update_store"),
    path('show_store_list', show_store_list, name="show_store_list"),

    path('show_store_info/<str:store_id>',
         show_store_info, name="show_store_info"),
    path('create_camera/<str:store_id>', create_camera, name="create_camera"),
]
