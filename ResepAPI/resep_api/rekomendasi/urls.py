from django.urls import path
from .views import rekomendasi_resep

urlpatterns = [
    path('rekomendasi/', rekomendasi_resep),
]
