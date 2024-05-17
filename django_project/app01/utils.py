import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import numpy as np
import pickle
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

#导入预训练模型
vgg16 = models.vgg16(pretrained=True)
vgg = vgg16.features # 获取vgg16的特征提取层(去除全连接层的卷积模块)
model_path = {
    'Swin Transformer': 'D:\pythonProject\monkey_classfication\myVGG17.pkl',
    'ConvNeXt': 'D:\pythonProject\monkey_classfication\myVGG.pkl',
    'XGBoost': 'D:\pythonProject\monkey_classfication\myVGG_952.pkl'
}

class CFG:
    image_size = 384
    class_names = ['X4_mean', 'X11_mean', 'X18_mean',
                   'X26_mean', 'X50_mean', 'X3112_mean', ]
    aux_class_names = list(map(lambda x: x.replace("mean", "sd"), class_names))
    num_classes = len(class_names)
    aux_num_classes = len(aux_class_names)


class PlantDataset(Dataset):
    def __init__(self, paths, features, labels=None, aux_labels=None, transform=None, augment=False):
        self.paths = paths
        self.features = features
        self.labels = labels
        self.aux_labels = aux_labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        feature = self.features[idx]

        # Read and decode image
        image = self.decode_image(path)

        # Apply augmentations
        if self.augment:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Ensure channel dimension is the first one
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)


        if self.labels is not None:
            label = torch.tensor(self.labels[idx])
            aux_label = torch.tensor(self.aux_labels[idx])
            return {'images': image, 'features': feature}, (label, aux_label)
        else:
            return {'images': image, 'features': feature}

    def decode_image(self, path):
        image = Image.open(path)
        image = image.resize((CFG.image_size,CFG.image_size))
        image = np.array(image)
        return image
class CustomModel(nn.Module):
    def __init__(self, num_classes, aux_num_classes, feature_cols, model_name='efficientnetv2_s'):
        super(CustomModel, self).__init__()

        # Define input layers
        self.img_input = nn.Identity()
        self.feat_input = nn.Identity()

        # Load pre-trained EfficientNetV2 model
        self.backbone = timm.create_model(model_name, pretrained=True)

        # Adapt the model to match the expected output size
        self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone.classifier = nn.Identity()

        self.dropout_img = nn.Dropout(0.2)

        # Branch for tabular/feature input
        self.dense1 = nn.Linear(len(feature_cols), 326)
        self.dense2 = nn.Linear(326, 64)
        self.dropout_feat = nn.Dropout(0.1)

        # Output layer
        self.head = nn.Linear(1064, num_classes)
        self.aux_head = nn.Linear(1064, aux_num_classes)

    def forward(self, img, feat):
        # Image branch
        x1 = self.backbone(img)
        x1 = self.dropout_img(x1.flatten(1))

        # Feature branch
        x2 = F.selu(self.dense1(self.feat_input(feat)))
        x2 = F.selu(self.dense2(x2))
        x2 = self.dropout_feat(x2)

        # Concatenate both branches
        concat = torch.cat([x1, x2], dim=1)
        # Output layer
        out1 = self.head(concat)
        out2 = F.relu(self.aux_head(concat))

        return {'head': out1, 'aux_head': out2}

def build_augmenter():
    # Define Albumentations augmentations
    transform = A.Compose([
        A.RandomBrightness(limit=0.1, always_apply=False, p=0.5),
        A.RandomContrast(limit=0.1, always_apply=False, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
        A.Cutout(num_holes=1, max_h_size=32, max_w_size=32, always_apply=False, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, p=0.5),
        ToTensorV2(),
    ])

    return transform

def build_dataset(paths, features, labels=None, aux_labels=None, batch_size=32, cache=True, augment=True, repeat=True, shuffle=1024, cache_dir="", drop_remainder=False):
    dataset = PlantDataset(paths, features, labels, aux_labels, transform=build_augmenter(), augment=augment)

    if cache_dir != "" and cache:
        os.makedirs(cache_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_remainder, pin_memory=True)

    return dataloader
def load_model(model_path, feature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomModel(CFG.num_classes, CFG.aux_num_classes, feature)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def process_data(image, feat, scaler_path):
    # 加载图片并应用预处理变换
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = image.convert('RGB')
    feat_values = np.array(feat).shape(-1, 1)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    standardized_feat = scaler.transform(feat_values)
    return transform(image).unsqueeze(0), standardized_feat  # 添加batch维度

def predict(image, feature, model='Swin Transformer'):
    path = model_path[model]
    model = load_model(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device, dtype=torch.float32)
    feature = feature.to(device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(image, feature)
    predictions = output['head'].cpu().numpy()


