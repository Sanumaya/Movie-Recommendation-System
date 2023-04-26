from django.shortcuts import render
from django.shortcuts import redirect, render
from . forms import *
from django.contrib import messages

# Create your views here.
def register(request):
    if request.method =="POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request,f'Aaccount has been created for{username}.Continue to login')
            return redirect('login')
    else:
        form = CreateUserForm()
    return render(request,'register.html',{'form':form})

def profile(request):
    return render(request,'profile.html')
