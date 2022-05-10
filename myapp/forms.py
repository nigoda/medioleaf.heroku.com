from django import forms
from django.contrib.auth.models import User
from myapp.models import UserProfileInfo,Image

class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput())


    class Meta():
        model = User
        fields = ('username','email','password')

class UserProfileInfoForm(forms.ModelForm):
    class Meta():
        model = UserProfileInfo
        fields = ('contact_number', 'profile_pic')

class ImageForm(forms.ModelForm):
 class Meta:
  model = Image
  fields = '__all__'
  labels = {'photo':''}
