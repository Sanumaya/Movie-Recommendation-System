from django.http.response import HttpResponse
from django.shortcuts import redirect, render
from diabetes.models import Result
import joblib
from .forms import ContactModelForm, ResultModelForm
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.core.mail import EmailMessage
from django.contrib import messages
from . models import Ans

# Create your views here.

def home(request):
    return render(request, 'home.html')

# def predict(request):
#     if request.method == "POST":
#         form = ResultModelForm(request.POST)
#         if form.is_valid():
#             form.save()
#             return redirect('home')
#         raise Exception('Invalid Input')
#     else:
#         form = ResultModelForm()
#         return render(request, 'predict.html', {'form':form})

def predict(request):
    print("predict view")
    if request.method =="POST":
        form = ResultModelForm(request.POST)
        print("post predict c=view ")
        x=request.POST
        for i in x:
            print(type(i))
            print(i)
        
        if form.is_valid():
            form.save()
            
    else:
        print("get predict view")
        form = ResultModelForm()
        return render(request, 'predict.html', context= {'form':form})
    return render(request, 'predict.html', context= {'form':form})

    


def result(request):
    cls = joblib.load('model.sav')
    lis = []
    lis.append(request.POST['pregnancies'])
    lis.append(request.POST['glucose'])
    lis.append(request.POST['blood_pressure'])
    lis.append(request.POST['skin_thickness'])
    lis.append(request.POST['insulin'])
    lis.append(request.POST['bmi'])
    lis.append(request.POST['diabetes_predigree_function'])
    lis.append(request.POST['age'])
    
    ans = round(cls.predict_proba([lis])[0][1])

    a = Ans(ans=ans,user = request.user)
    a.save()


    return render(request, 'result.html', {'ans':ans})


def about(request):
    return render(request,'about.html')


def contact(request):
    if request.method == "GET":
        form = ContactModelForm()
        return render(request,'contact.html', {'form':form})
    else:   
        form = ContactModelForm(request.POST)
        if form.is_valid():
            form.save()
            print(form.cleaned_data)
            message = form.cleaned_data['description']
            from_email = form.cleaned_data['mail']
            phone = form.cleaned_data['phone']
            message = message+str(phone)
            try:
                send_mail(
                    'Contact us form message',
                    message,
                    
                    from_email,
                    ['dpsystem99@gmail.com'],
                    fail_silently=False,
                )
                
            except Exception as e:
                print(str(e))
            messages.success(request, 'Successfully sent you message to HPS in gmail')
            
            return redirect('home')


# def record(request): #Diplays previous record of authenticated user
#     if request.user.is_authenticated:
#         # record_data = Result.objects.filter(owner = request.user) #Filter only those data whose owner is the logged in user
#         # print(record_data)
#         # return render(request , 'record.html' , {'record_data':record_data})

#     return redirect('home')
            
            
    


