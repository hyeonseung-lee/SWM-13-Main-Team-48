from django.shortcuts import render
from django.contrib.auth import get_user_model
from ai_cctv.settings import *
from django.shortcuts import redirect
import requests
from json import JSONDecodeError
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate,logout,login
from django.contrib import messages
import jwt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
import hashlib

# Create your views here.
User=get_user_model()

def kakao_login(request):
    CLIENT_ID = SOCIAL_OUTH_CONFIG['KAKAO_REST_API_KEY']
    REDIRET_URL = SOCIAL_OUTH_CONFIG['KAKAO_REDIRECT_URI']
    state = "none"
    url = "https://kauth.kakao.com/oauth/authorize?response_type=code&client_id={0}&redirect_uri={1}&respons_type=code&state={2}".format(
        CLIENT_ID, REDIRET_URL, state)
    res = redirect(url)
    
    return res


# Refresh_token, Access_token 받기

@api_view(['GET'])
def get_token(request):
    # state = request.GET.get("state")
    CODE = request.query_params['code']
    url = "https://kauth.kakao.com/oauth/token"
    req = {
            'grant_type': 'authorization_code',
            'client_id': SOCIAL_OUTH_CONFIG['KAKAO_REST_API_KEY'],
            'redirect_url': SOCIAL_OUTH_CONFIG['KAKAO_REDIRECT_URI'],
            'client_secret': SOCIAL_OUTH_CONFIG['KAKAO_SECRET_KEY'],
            'code': CODE
        }
    headers = {
        'Content-type': 'application/x-www-form-urlencoded;charset=utf-8'
    }

    response = requests.post(url, data=req, headers=headers)
    tokenJson = response.json()

    # print(tokenJson)
    try:
        userUrl = "https://kapi.kakao.com/v2/user/me" # 유저 정보 조회하는 uri
        auth = "Bearer "+ tokenJson['access_token'] ## 'Bearer '여기에서 띄어쓰기 필수!!
        HEADER = {
            "Authorization": auth,
            "Content-type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        res = requests.get(userUrl, headers=HEADER).json()
        # 처음 회원가입 하는경우
        if not User.objects.filter(id=res['id']).exists():
            # 유저 생성
            password=hashlib.sha256(str(res['id']).encode()).hexdigest()
            user=User.objects.create_user(id=res['id'],password=password)
            # print(password)
            print("처음 회원가입 하는경우, 바로 로그인")
            login(request,user)

            # 여기에서 사용자 정보 받는 곳으로 Redirect 시켜야함 -----------------------
            return redirect('users:profile_update_page')
        # 회원가입이 되어있는 경우 
        else:
            print("이미 회원가입은 했음")
            password=hashlib.sha256(str(res['id']).encode()).hexdigest()
            user=authenticate(id=res['id'],password=password)
            # print(user)
            # print("무조건 되어야함")
            if user is not None:
                login(request,user)
                return redirect('dashboards')
            else:
                messages.warning(request, "비밀번호가 틀렸거나 회원이 아닙니다.")
                return redirect('dashboards')
    except KeyError:
        messages.warning(request, "키가 에러가 났습니다.")
        return redirect('dashboards')

    except JSONDecodeError:
        messages.warning(request, "디코더 에러가 났습니다.")
        return redirect('dashboards')

    except jwt.DecodeError:
        messages.warning(request, "카카오 jwt 에러가 났습니다.")
        return redirect('dashboards')

    except ConnectionError:
        messages.warning(request, "연결 오류 에러가 났습니다.")
        return redirect('dashboards')

# def profile_update(request):
#     return redirect('dashboards')

def service_logout(request):
    logout(request)
    print('장고 로그아웃')
    return redirect('users:kakao_logout')

# Web용 로그아웃
def kakao_logout(request):
    CLIENT_ID = SOCIAL_OUTH_CONFIG['KAKAO_REST_API_KEY']
    KAKAO_LOGOUT_REDIRECT_URI = SOCIAL_OUTH_CONFIG['KAKAO_LOGOUT_REDIRECT_URI']
    url = "https://kauth.kakao.com/oauth/logout?&client_id={0}&logout_redirect_uri={1}".format(
        CLIENT_ID, KAKAO_LOGOUT_REDIRECT_URI)
    # print(KAKAO_LOGOUT_REDIRECT_URI)
    res = redirect(url)
    return res
    # print('카카오 로그아웃')
def go_main(request):
    return redirect('dashboards')

def profile_update_page(request):
    return render(request,'profile_update_page.html')

def profile_update(request):
    try:
        request.user.profile.username=request.POST['username']
        request.user.profile.photo=request.FILES.get('photo')
        request.user.profile.save()
        return redirect('dashboards')

    except:
        messages.warning(request, "사용자 이름 or 사진이 없습니다")
        return redirect('users:profile_update_page')

