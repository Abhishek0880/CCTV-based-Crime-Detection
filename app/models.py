from django.db import models

# Create your models here.
class UploadedVideo(models.Model):
    video = models.FileField()
    prediction = models.CharField(max_length=100)

    def __str__(self) -> str:
        return super().__str__()