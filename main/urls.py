from django.urls import path
from . import views

app_name = 'main'

urlpatterns = [
    path('', views.index, name='index'),
    path('du-lieu/', views.data, name='data'),
    path('du-doan/', views.predict, name='predict'),
    path('tai-khoan/', views.account, name='account'),
    path('api/dashboard-csv/', views.dashboard_csv_data, name='dashboard_csv_data'),
] 