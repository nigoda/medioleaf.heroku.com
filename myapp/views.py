from django.shortcuts import render
from myapp.forms import UserProfileInfoForm, UserForm,ImageForm
from .models import Image
from django.contrib.auth import authenticate, login, logout
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.contrib.auth.decorators import login_required
import os
import shutil
from myapp.test import test


def index(request):
    return render(request, 'myapp/index.html')

def dashboard(request):
    return render(request, 'myapp/dashboard.html')

@login_required
def user_logout(request):
    logout(request)
    return HttpResponseRedirect(reverse('index'))

@login_required
def special(request):
    return HttpResponse("You are logged in, Nice!")

def register(request):

    registered = False

    if request.method == "POST":
        user_form = UserForm(data=request.POST)
        profile_form = UserProfileInfoForm(data=request.POST)

        if user_form.is_valid() and profile_form.is_valid():

            user = user_form.save()
            user.set_password(user.password)
            user.save()

            profile = profile_form.save(commit=False)
            profile.user = user

            if 'profile_pic' in request.FILES:
                profile.profile_pic = request.FILES['profile_pic']

            profile.save()

            registered = True
        else:
            print(user_form.errors, profile_form.errors)

    else:
        user_form = UserForm()
        profile_form = UserProfileInfoForm()

    return render(request,'myapp/registraction.html',
                            {'user_form': user_form,
                              'profile_form':profile_form,
                              'registered':registered})
def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')


        user = authenticate(username=username, password=password)

        if user:
            if user.is_active:
                login(request,user)
                return HttpResponseRedirect(reverse('index'))
            else:
                return HttpResponse("ACCOUNT NOT ACTIVE")

        else:
            print("Someone tried to login and failed!")
            print("Username: {} and Password {}".format(username,password))
            return render(request, 'myapp/loginfail.html',{})
    else:
        return render(request, 'myapp/login.html',{})


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def dashboard(request):
 text = "Choose the Image of leaf"
 if request.method == "POST":
  form = ImageForm(request.POST, request.FILES)
  ph = float(request.POST.get('ph'))
  tem = float(request.POST.get('temparature'))
  text = test("test.jpg",ph,tem)
  if text[0]:
      text = "Medicinal"
  else:
      text = "Non-Medicinal"

  if form.is_valid():
   form.save()
   os.chdir(BASE_DIR+"/media/myimage")
   result = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime)
   image = result[len(result)-1] #image path
   original = BASE_DIR + "/media/myimage/" + image
   target = BASE_DIR + "/static/myapp/images/test.jpg"
   shutil.move(original, target) # replace move to copy to save upladed images in database
   print(image)
 form = ImageForm()
 img = Image.objects.all()
 return render(request, 'myapp/dashboard.html', {'img':img, 'text':text, 'form':form})
