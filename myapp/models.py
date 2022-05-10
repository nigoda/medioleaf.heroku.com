from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class UserProfileInfo(models.Model):


    user = models.OneToOneField(User,on_delete=models.CASCADE)

    #Additional
    contact_number = models.CharField(max_length=256,unique=True)

    profile_pic = models.ImageField(upload_to='profile_pic', blank=True)


    def __str__(self):
        return self.user.username

class Image(models.Model):
 photo = models.ImageField(upload_to="myimage")
 date = models.DateTimeField(auto_now_add=True)
 ph = models.FloatField()
 temparature = models.FloatField()
 #user user@gmail.com user123
