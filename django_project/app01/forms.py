# forms.py
from django import forms

from .models import ImageUpload


class DocumentForm(forms.Form):
    document = forms.FileField()

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ImageUpload
        fields = ['image']

class UploadForm(forms.Form):
    image_upload = forms.FileField(label='选择图片')
    file_upload = forms.FileField(label='选择文件')
    choice_model = forms.ChoiceField(choices=[
        ('Swin Transformer', 'Swin Transformer'),
        ('ConvNeXt', 'ConvNeXt'),
        ('XGBoost', 'XGBoost')
    ], label='选择模型')