from django.db import models


# Create your models here.
class Info(models.Model):
    id = models.IntegerField('ID', primary_key=True)
