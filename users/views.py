from django.shortcuts import render
from django.contrib.auth import get_user_model
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import *
from ai_cctv.settings import *
from django.shortcuts import redirect
import requests
from rest_framework.response import Response
from rest_framework import status
from json import JSONDecodeError
import jwt
from django.shortcuts import get_object_or_404

# Create your views here.
User=get_user_model()
   

@api_view(['GET'])
@permission_classes([AllowAny])
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
@permission_classes([AllowAny])
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

        if not User.objects.filter(id=res['id']).exists():
            serializer=UserSerializer(data=res)
            if serializer.is_valid(raise_exception=True):
                token=serializer.save()
                return Response(token,status=status.HTTP_201_CREATED) # 회원가입 페이지로 넘어가야함 -> json을 update 형태로 보냄
        # 회원가입이 되어있는 경우 
        else:
            user = get_object_or_404(User,id=res['id'])
            refresh = RefreshToken.for_user(user)
            token={"refresh":str(refresh),"access":str(refresh.access_token)}
            return Response(token,status=status.HTTP_202_ACCEPTED) # 메인페이지로
    except KeyError:
        return Response({'message': 'KEY_ERROR'}, status=status.HTTP_400_BAD_REQUEST)
    
    except JSONDecodeError:
        return Response({'message': 'JSON_DECODE_ERROR'}, status=status.HTTP_400_BAD_REQUEST)

    except jwt.DecodeError:
        return Response({'message': 'JWT_DECODE_ERROR'}, status=status.HTTP_400_BAD_REQUEST)

    except ConnectionError:
        return Response({'message': 'CONNECTION_ERROR'}, status=status.HTTP_400_BAD_REQUEST)

# 추가 회원 가입 정보 업데이트
# 헤더에 Bearer token(access_token) 담고 json 으로 업데이트 정보 보내주기 -> 되면 201로 응답  
@api_view(['PUT'])
def profile_update(request):
    data = json.loads(request.body)

    if data['job']=='중상':
        data['company_place_id']=request.user.id
    # 식당 외에는 company를 판별할 값이 없어서 company_place_id에 카카오 Id를 담아두기

    company_serializer=CompanySerializer(data=data)

    # 여기서 1차로 company 테이블에 해당 회사가 있는지 확인
    # company_code, company_place_id는 unique값이어야해서 지도에서 가져온값이나 해당 1인회사의 경우 else로 Update 부분으로 넘어감
    if company_serializer.is_valid():
        company=company_serializer.save()
        data["company"]=company["company"]
    else:
        exist=get_object_or_404(Company,company_place_id=data['company_place_id'])       
        company_serializer=CompanySerializer(exist,data=data)
        # 업데이트에서는 unique값들은 업데이트하지않음(이 값들이 다르면 새로운 회사인거임)
        if company_serializer.is_valid(raise_exception=True):
            company=company_serializer.save()
            data["company"]=company.company_code

    # 기존으로 user가 생성되며 post_save로 profile이 생성됨, 그래서 초기 세팅일때도 update로 해줘야함
    # 개발 테스트용으로 회사가 새로생기거나 job이 바뀌는 경우 아래로 profile update
    serializer=ProfileSerializer(request.user.profile,data=data)
    if serializer.is_valid(raise_exception=True):
        token=serializer.save()

    return Response(token,status=status.HTTP_201_CREATED) ## 메인 페이지로


# Web용 로그아웃
@api_view(['GET'])
@permission_classes([AllowAny])
def kakao_logout(request):
    CLIENT_ID = SOCIAL_OUTH_CONFIG['KAKAO_REST_API_KEY']
    KAKAO_LOGOUT_REDIRECT_URI = SOCIAL_OUTH_CONFIG['KAKAO_LOGOUT_REDIRECT_URI']
    url = "https://kauth.kakao.com/oauth/logout?&client_id={0}&logout_redirect_uri={1}".format(
        CLIENT_ID, KAKAO_LOGOUT_REDIRECT_URI)
    res = redirect(url)

    return res

#슈퍼유저 사용
@permission_classes([AllowAny]) 
class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class=MyTokenObtainPairSerializer
