from pickle import TRUE
from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Result(models.Model):
    pregnancies = models.CharField(max_length=15)
    glucose = models.CharField(max_length=15)
    blood_pressure = models.CharField(max_length=15)
    skin_thickness = models.CharField(max_length=15)
    insulin = models.CharField(max_length=15)
    bmi = models.CharField(max_length=15)
    diabetes_predigree_function = models.CharField(max_length=15)
    age = models.CharField(max_length=15)
    owner = models.ForeignKey(User,on_delete=models.CASCADE,null=TRUE)

class Contact(models.Model):
    name = models.CharField(max_length=50)
    mail = models.EmailField()
    phone = models.IntegerField()
    description = models.TextField()

    def __str__(self):
        return self.name

class Ans(models.Model):
    ans = models.CharField(max_length=10)
    user = models.ForeignKey(User,on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.user.username



