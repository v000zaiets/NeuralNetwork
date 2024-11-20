from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from neuralnet.views import upload_and_predict  # Import your view

urlpatterns = [
    path('', upload_and_predict, name='upload_and_predict'),  # Set the upload page as the home page
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
