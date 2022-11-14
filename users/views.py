from django.shortcuts import render, redirect
from django.contrib.auth import get_user_model
from ai_cctv.settings import *
import requests
from json import JSONDecodeError
from django.shortcuts import get_object_or_404
from django.contrib.auth import authenticate, logout, login
from django.contrib import messages
import jwt
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
import hashlib
from django.contrib.auth.decorators import login_required
from .models import *
# Create your views here.
User = get_user_model()

# 카카오 로그인


def kakao_login(request):
    CLIENT_ID = SOCIAL_OUTH_CONFIG['KAKAO_REST_API_KEY']
    REDIRET_URL = SOCIAL_OUTH_CONFIG['KAKAO_REDIRECT_URI']
    state = "none"
    url = "https://kauth.kakao.com/oauth/authorize?response_type=code&client_id={0}&redirect_uri={1}&respons_type=code&state={2}".format(
        CLIENT_ID, REDIRET_URL, state)
    res = redirect(url)

    return res


# 카카오 Refresh_token, Access_token 받기 + 장고 로그인
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
        userUrl = "https://kapi.kakao.com/v2/user/me"  # 유저 정보 조회하는 uri
        auth = "Bearer " + tokenJson['access_token']  # 'Bearer '여기에서 띄어쓰기 필수!!
        HEADER = {
            "Authorization": auth,
            "Content-type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        res = requests.get(userUrl, headers=HEADER).json()
        # 처음 회원가입 하는경우
        if not User.objects.filter(id=res['id']).exists():
            # 유저 생성
            password = hashlib.sha256(str(res['id']).encode()).hexdigest()
            user = User.objects.create_user(id=res['id'], password=password)
            # print(password)
            print("처음 회원가입 하는경우, 바로 로그인")
            login(request, user)

            # 여기에서 사용자 정보 받는 곳으로 Redirect 시켜야함 -----------------------
            return redirect('users:profile_update_page')
        # 회원가입이 되어있는 경우
        else:
            print("이미 회원가입은 했음")
            password = hashlib.sha256(str(res['id']).encode()).hexdigest()
            user = authenticate(id=res['id'], password=password)
            # print(user)
            # print("무조건 되어야함")
            if user is not None:
                login(request, user)
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

# Web용 로그아웃


@login_required(login_url='dashboards')
def service_logout(request):
    logout(request)
    print('장고 로그아웃')
    return redirect('users:kakao_logout')

# 카카오 로그아웃


def kakao_logout(request):
    CLIENT_ID = SOCIAL_OUTH_CONFIG['KAKAO_REST_API_KEY']
    KAKAO_LOGOUT_REDIRECT_URI = SOCIAL_OUTH_CONFIG['KAKAO_LOGOUT_REDIRECT_URI']
    url = "https://kauth.kakao.com/oauth/logout?&client_id={0}&logout_redirect_uri={1}".format(
        CLIENT_ID, KAKAO_LOGOUT_REDIRECT_URI)
    # print(KAKAO_LOGOUT_REDIRECT_URI)
    res = redirect(url)
    return res
    # print('카카오 로그아웃')

# ---- 카카오에서 리다이랙트 후 main으로 redirect ----


def go_main(request):
    return redirect('dashboards')

# ---------- 프로필 업데이트 페이지(이름, 사진) -------------


@login_required(login_url='dashboards')
def profile_update_page(request):
    return render(request, 'profile_update_page.html')

# ---------- 프로필 업데이트(이름, 사진) -------------


@login_required(login_url='dashboards')
def profile_update(request):
    try:
        print(request.POST)
        request.user.profile.username = request.POST['username']
        if request.FILES.get('photo'):
            request.user.profile.photo = request.FILES.get('photo')
        request.user.profile.save()
        return redirect('profile')
    except:
        messages.warning(request, "사용자 이름 or 사진이 없습니다")
        return redirect('profile')


# ---------- 매장 생성 페이지 -------------
@login_required(login_url='dashboards')
def create_store_page(request):
    return render(request, 'store/store_page.html')

# ---------- 매장 생성 -------------


@login_required(login_url='dashboards')
def create_store(request):
    Store.objects.create(
        name=request.POST['name'],
        address=request.POST['address'],
        owner=request.user,
    )
    return redirect('users:show_store_list')

# ---------- 매장 정보 업데이트 페이지 -------------


@login_required(login_url='dashboards')
def update_store_page(request, store_id):
    store = Store.objects.get(id=store_id)
    return render(request, 'store/store_page.html', {"store": store})

# ---------- 매장 정보 업데이트 -------------


@login_required(login_url='dashboards')
def update_store(request, store_id):
    store = Store.objects.get(id=store_id)
    store.name = request.POST['name']
    store.address = request.POST['address']
    store.save()
    return redirect('users:show_store_list')

# ---------- 매장 삭제 -------------


@login_required(login_url='dashboards')
def delete_store(request, store_id):
    store = Store.objects.get(id=store_id)
    store.delete()
    return redirect('users:show_store_list')

# ---------- 매장 리스트 열람 -------------


@login_required(login_url='dashboards')
def show_store_list(request):
    storelist = Store.objects.filter(owner=request.user)

    return render(request, 'store/store_list.html', {"storelist": storelist})

# ---------- 매장 메인으로 설정 -------------


@login_required(login_url='dashboards')
def set_main_store(request, store_id):

    # 있든 말든 객체를 바꿔주면되니까 상관없음
    store = Store.objects.get(id=store_id)
    request.user.profile.main_store = store
    request.user.profile.save()

    return redirect('users:show_store_list')


# ---------- 매장 및 카메라 정보 열람 -------------
@login_required(login_url='dashboards')
def show_store_info(request, store_id):
    store = Store.objects.get(id=store_id)
    cameras = Camera.objects.filter(store=store)
    default = Camera.objects.filter(store=store, main_cam=True)
    result = None
    if default.exists():
        result = default.first()
    default = default.exists()

    context = {
        "store": store,
        "cameras": cameras,
        "is_default": default,
        "result": result
    }
    return render(request, 'store/store_and_camera_info.html', context)

# ---------- 카메라 생성 -------------


@login_required(login_url='dashboards')
def create_camera(request, store_id):
    store = Store.objects.get(id=store_id)

    Camera.objects.create(
        rtsp_url=request.POST['rtsp_url'],
        store=store
    )
    return redirect('users:', store_id)


# ---------- 카메라 주소 업데이트 페이지 -------------
@login_required(login_url='dashboards')
def update_camera_page(request, store_id, camera_id):
    camera = Camera.objects.get(id=camera_id)
    context = {
        "store_id": store_id,
        "camera": camera
    }
    return render(request, 'camera/update_camera_page.html', context)

# ---------- 카메라 주소 업데이트 -------------


@login_required(login_url='dashboards')
def update_camera(request, store_id, camera_id):
    camera = Camera.objects.get(id=camera_id)
    camera.rtsp_url = request.POST['rtsp_url']
    camera.save()
    return redirect('users:show_store_info', store_id)


# ---------- 카메라 삭제 -------------
@login_required(login_url='dashboards')
def delete_camera(request, store_id, camera_id):
    camera = Camera.objects.get(id=camera_id)
    camera.delete()
    return redirect('users:show_store_info', store_id)

# ---------- default 카메라 선정 : 가게가 메인 가게일때만 가능 -------------


@login_required(login_url='dashboards')
def set_default_camera(request, store_id, camera_id):
    store = Store.objects.get(id=store_id)
    if request.user.profile.main_store == store:
        default = Camera.objects.filter(store=store, main_cam=True)
        if default.exists():
            default = default.first()
            default.main_cam = False  # 무조건 한개
            default.save()

        camera = Camera.objects.get(id=camera_id)
        camera.main_cam = True
        camera.save()
        return redirect('users:show_store_info', store_id)

    else:
        messages.warning(request, "해당가게는 메인 가게가 아닙니다")
        return redirect('users:show_store_info', store_id)
