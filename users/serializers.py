from dataclasses import fields
from rest_framework import serializers
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.tokens import RefreshToken, TokenError
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from users.models import *

User=get_user_model() #커스텀 유저 가져옴 
      
class UserSerializer(serializers.ModelSerializer):
    
    # profile = ProfileSerializer(required=True)
    
    class Meta:
        model=User
        # fields=['id','profile']
        fields=['id']
        
        
    def create(self,validated_data): # view에서 serializer save함수 호출하면 create 또는 perform_create(생성) 가 호출됨 
        user = User.objects.create(
            id = validated_data["id"]
        )
        user.save()
        
        refresh = RefreshToken.for_user(user)
        return {"refresh":str(refresh),"access":str(refresh.access_token)}

# superuser용
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self,attrs):
        data = super().validate(attrs)
        refresh=self.get_token(self.user)
        data["refresh"]=str(refresh)
        data["access"]=str(refresh.access_token)        
       
        return data
