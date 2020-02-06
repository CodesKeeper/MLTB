"""MLTB URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from theory import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('theory/', include('theory.urls')),  # 算法原理
    path('performance/', include('performance.urls')),  # 算法性能分析
    path('example/', include('example.urls')),  # 算法应用
    path('index/', views.index, name='index'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('register/', views.register, name='register'),
    path('document/', views.document, name='document'),  # 文档
    path('about/', views.about, name='about'),
]
