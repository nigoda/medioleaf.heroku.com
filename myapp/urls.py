from django.urls import path
from myapp import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'myapp'

urlpatterns=[
    path('register/', views.register, name='register'),
    path('user_login/', views.user_login, name='user_login'),
    path('dashboard/', views.dashboard, name='dashboard'),
    ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
