import os
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
import torchvision.transforms.functional as transF
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm

from torchvision import models
from typing import Optional, List
import shutil
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.loss import RSquare
from utils.my_dataset import MyEvalDataSet
from model import swin_base_patch4_window12_384_in22k as swin_transformer
from model import linear_encoder, linear_decoder
from tqdm import tqdm
from typing import List
from torch import nn
import logging
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2


def tensorToNdImg(tensor):
    ndRes = tensor.squeeze()
    ndRes = (
        (ndRes - torch.min(ndRes)) / (torch.max(ndRes) - torch.min(ndRes) + 1e-8) * 255
    )
    ndRes = ndRes.detach().cpu().numpy()
    return ndRes


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return tensor * std + mean


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))

    # load pretrained model
    # IFCNNmodel = IFCNN().to(device)
    # weightFile = os.path.join('checkPoints'+'IFCNN','IFCNN-MAX.pth')
    # IFCNNmodel.load_state_dict(torch.load(weightFile))
    # IFCNNmodel.eval()
    savemodelName = "model_epoch54"
    image_encoder = swin_transformer().to(device)
    number_encoder = linear_encoder().to(device)
    decoder = linear_decoder(in_channel=1350, out_channel=6, hidden_channel=256).to(
        device
    )

    weightFile = os.path.join("pretrained", savemodelName + ".tar")
    chkPoint = torch.load(weightFile)
    print("load weightFile:{}".format(weightFile))
    if torch.cuda.device_count() > 1:
        image_encoder.module.load_state_dict(chkPoint["image_encoder_state_dict"])
    else:
        image_encoder.load_state_dict(chkPoint["image_encoder_state_dict"])
    number_encoder.load_state_dict(chkPoint["num_encoder_state_dict"])
    decoder.load_state_dict(chkPoint["decoder_state_dict"])
    image_encoder.eval()
    number_encoder.eval()
    decoder.eval()

    data_root = os.path.abspath(os.path.join(os.getcwd(), "data"))
    train_root = os.path.join(data_root, "train")
    testRoot = os.path.join(data_root, "test")
    test_trans =A.Compose(
        [
            A.Resize(224, 224),
            A.ToFloat(),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225],max_pixel_value=1.0),
            ToTensorV2(),
        ]
    )
    test_dataset = MyEvalDataSet(
        csv_path=os.path.join(testRoot, "test.csv"), img_transform=test_trans
    )
    predict_df = pd.DataFrame(columns=["id", "X4", "X11", "X18", "X50", "X26", "X3112"])

    for index, (
        img,
        aux,
        id,
    ) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        img = img.to(device, non_blocking=True).unsqueeze(0)
        aux = aux.to(device, non_blocking=True).unsqueeze(0)
        # perform image fusion
        with torch.no_grad():
            img_feature = image_encoder(img)
            num_feature = number_encoder(aux)
            out = decoder(torch.cat((img_feature, num_feature), dim=1))
        out = out.squeeze().cpu().detach().numpy()
        out[3], out[4] = out[4], out[3]
        delog_out = np.exp(out)
        predict_df.loc[len(predict_df)] = [id] + list(delog_out)

    predict_df.to_csv(os.path.join(data_root, "predict.csv"), index=False)


if __name__ == "__main__":
    test()
