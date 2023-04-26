from django import forms
from .models import Result, Contact


class ResultModelForm(forms.ModelForm):
    class Meta:
        model = Result
        fields = "__all__"

class ContactModelForm(forms.ModelForm):
    class Meta:
        model = Contact
        fields = "__all__"
        