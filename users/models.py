from django.db import models
from django.contrib.auth.models import BaseUserManager, AbstractBaseUser, PermissionsMixin
from django.utils.translation import gettext_lazy as _
from django.dispatch import receiver
from django.db.models.signals import post_save

# Create your models here.

class UserManager(BaseUserManager):
    def create_user(self, id, password):  # user 생성 함수
        user = self.model(
            id=id
        )
        user.set_password(password)

        user.save(using=self._db)  # settings에 db중 기본 db 사용한다는 의미
        return user

    def create_superuser(self, id, password):  # superuser 생성 함수
        user = self.create_user(
            id=id
        )
        user.set_password(password)

        user.is_superuser = True
        user.is_admin = True
        user.is_staff = True
        user.save(using=self._db)
        return user
# is_active(일반사용자)랑 is_admin은 장고 유저 모델의 필수필드라 정의
# is_staff(사이트관리스탭->이 값이 true여야 관리자페이지 로그인 가능)
# is_superuser는 관리자 페이지의 내용을 제한없이 봄

# PermissionMixin: admin계정 로그인시 사용자 권한 요구하는데 그때 해결
# AbstractBaseUser는 기존필드 만족안하고 완전히 새로운 모델 생성할때


class User(AbstractBaseUser, PermissionsMixin):
    id = models.CharField(primary_key=True, unique=True, max_length=100)

    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)
    is_admin = models.BooleanField(default=False)
    is_staff = models.BooleanField(default=False)

    objects = UserManager()

    USERNAME_FIELD = 'id'  # 고유식별자


class Store(models.Model):
    name=models.CharField(max_length=200,null=True,blank=True) 
    address=models.CharField(max_length=100,null=True,blank=True) # 주소
    owner=models.ForeignKey(User,on_delete=models.CASCADE,db_column='owner',related_name='stores')
    current_visitor=models.IntegerField(default=0,null=True,blank=True)


# related_name은 역참조시에 _set 대신 사용
class Camera(models.Model):
    # 같은 모델에 대해 onetoone, foreignkey 두개 다 설정할때 역참조에서 구별이 안됨을 방지하기위해(외래키에서 _set으로 구별이 되지만 에러가나긴함) related_name을 적어줘야함
    store=models.ForeignKey(Store, on_delete=models.CASCADE, null=True, blank=True,related_name='cameras', db_column='store') 
    rtsp_url=models.CharField(max_length=100,null=True, blank=True) 
    default_cam = models.BooleanField(default=False,null=True,blank=True)

class Profile(models.Model):
    user = models.OneToOneField(User,on_delete=models.CASCADE,db_column='user')
    main_store = models.OneToOneField(Store,on_delete=models.SET_NULL,null=True,blank=True)
    fcm_token=models.CharField(max_length=200,null=True,blank=True,unique=True) # fcm push token (id)
    photo=models.ImageField(upload_to='user_photo/',null=True,blank=True)
    username=models.CharField(max_length=50,null=True,blank=True) 
    # 몇명 방문한지 기록
#     # app_push_check=models.BooleanField(verbose_name=_("앱 푸시"), default=True) # 앱 push 알림 선택
#     # email_push_check=models.BooleanField(verbose_name=_("email 푸시"),default=False) # email push 알림




@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()
