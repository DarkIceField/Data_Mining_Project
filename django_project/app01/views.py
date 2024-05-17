from PIL import Image
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, HttpResponse, redirect
from .forms import DocumentForm, ImageUploadForm, UploadForm
from .models import Document, ImageUpload

from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
# Create your views here.
import datetime

def times(request):
    now = datetime.datetime.now().strftime("%Y-%m-%d %X")
    return render(request, "timer.html", {"now": now})
def main(request):
    if request.method == 'POST' and request.FILES.get('image_upload', None):
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image_upload']
            feat = request.FILES['file_upload']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            uploaded_image_url = fs.url(filename)
            model = form.cleaned_data['choice_model']
            image = Image.open(image)
            #predicted_label, probs = predict(image, model)
            response = {
                'success': True,
                'image_url': uploaded_image_url,
                'table': {
                    'data1': 1,
                    'data2': 2,
                    'data3': 3,
                    'data4': 4,
                    'data5': 5,
                    'data6': 6,
                    'predicted_label': 0,
                    'choice_model': model
                }

            }
            return JsonResponse(response)
    return render(request, 'main.html')
def home(request):

    if request.method == 'POST' and request.FILES['image']:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.save()

            # 获取上传的图片URL
            image_url = instance.image.url
            image = request.FILES['image_upload']
            image = Image.open(image)
            predicted_label, probs = predict(image)
            response_data = {
                'success': True,
                'image_url': predicted_label,
            }
            # 如果你希望返回JSON数据给前端
        else:
            response_data = {
                'success': False,
            }
        return JsonResponse(response_data)

    data = {
        'data1': 1,
        'data2': 2,
        'data3': 3,
        'data4': 4,
        'data5': 5,
        'data6': 6

    }
    return render(request, 'home.html', data)

def upload_predict(request):
    if request.method == 'POST' and request.FILES['image']:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.save()

            # 获取上传的图片URL
            image_url = instance.image.url
            image = request.FILES['image']
            image = Image.open(image)
            predicted_label, probs = predict(image)
            response_data = {
                'success': True,
                'predicted_label': predicted_label,
                'probabilities': probs,
                'image_url': image_url  # 如果需要，你可以添加图片URL
            }
            # 如果你希望返回JSON数据给前端
            return JsonResponse(response_data)

        else:
            # 表单验证失败，处理错误
            return render(request, 'upload_predict.html', {'form': form, 'errors': form.errors})

    else:
        form = ImageUploadForm()
        # 渲染模板，传递表单
        return render(request, 'upload_predict.html', {'form': form})

def image_list(request):
    images = ImageUpload.objects.all()
    return render(request, 'image_list.html', {'images': images})

def upload_show_image(request):
    if request.method == 'POST' and request.FILES['image']:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.save()

            # 获取上传的图片URL
            image_url = instance.image.url

            # 如果你希望返回JSON数据给前端
            return JsonResponse({'image_url': image_url})

        else:
            # 表单验证失败，处理错误
            pass

    else:
        form = ImageUploadForm()

        # 渲染模板，传递表单
    return render(request, 'upload_form.html', {'form': form})


def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']
        image_upload = ImageUpload(image=image)  # 这里假设ImageUpload模型有一个FileField叫image
        image_upload.save()  # 这将保存对象并自动处理文件保存到MEDIA_ROOT
        
        uploaded_image = ImageUpload.objects.latest('id')
        return redirect('show_image', uploaded_image.id)
    form = ImageUploadForm()
    return render(request, 'upload_image.html', {'form': form})

def show_image(request, image_id):
    # 根据ID获取图片对象  
    uploaded_image = ImageUpload.objects.get(id=image_id)
    return render(request, 'show_image.html', {'image': uploaded_image})

def upload_file(request):
    if request.method == 'POST' and request.FILES['document']:
        document = request.FILES['document']
        if document:
            # 如果需要存储文件到数据库
            new_doc = Document(document=document)
            new_doc.save()
            # 重定向到另一个视图或页面
            return HttpResponseRedirect(reverse('success_upload'))
    form = DocumentForm()
    return render(request, 'upload.html', {'form': form})