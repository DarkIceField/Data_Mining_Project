import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.loss import RSquare
from utils.my_dataset import MyDataSet
from model import swin_base_patch4_window7_224_in22k as swin_transformer
from model import linear_encoder
from model import linear_decoder
from tqdm import tqdm
from typing import List
from torch import nn
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


def saveCheckPoint(
    epoch: int,
    image_encoder: nn.Module,
    num_encoder: nn.Module,
    decoder: nn.Module,
    image_optimizer: optim.Optimizer,
    other_optimizer: optim.Optimizer,
    image_scheduler: optim.lr_scheduler.LRScheduler,
    other_scheduler: optim.lr_scheduler.LRScheduler,
    loss: torch.Tensor,
    weightDir: str,
):
    torch.save(
        {
            "epoch": epoch,
            "image_encoder_state_dict": image_encoder.state_dict(),
            "num_encoder_state_dict": (
                num_encoder.module.state_dict()
                if torch.cuda.device_count() > 1
                else num_encoder.state_dict()
            ),
            "decoder_state_dict": decoder.state_dict(),
            "loss": loss,
            "image_optimizer_state_dict": image_optimizer.state_dict(),
            "other_optimizer_state_dict": other_optimizer.state_dict(),
            "scheduler1_state_dict": image_scheduler.state_dict(),
            "scheduler2_state_dict": other_scheduler.state_dict(),
        },
        os.path.join(weightDir, "model_epoch{}.tar".format(epoch)),
    )


def getDualOutputLogger(logFile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%Y-%M-%d %H:%M:%S"
    )
    fh = logging.FileHandler(logFile)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def validate(
    image_encoder: nn.Module,
    number_encoder: nn.Module,
    decoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    epoch: int,
    logger: logging.Logger,
):
    r2loss = RSquare(is_loss=True).to(device)
    lossSum = 0.0
    print("start test epoch {}:".format(epoch))

    with torch.no_grad():
        for index, (img, aux, label) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            img = img.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            img_feature = image_encoder(img)
            num_feature = number_encoder(aux)
            out = decoder(torch.cat((img_feature, num_feature), dim=1))
            loss = r2loss(out, label)
            lossSum += loss.item()

    loss_mean = lossSum / len(dataloader)
    logger.info("end epoch{} test,vali_loss:{:.4f},\n".format(epoch, loss_mean))

    return loss_mean


def train():
    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    tb_writer = SummaryWriter()

    # set parameters
    data_root = os.path.abspath(os.path.join(os.getcwd(), "data"))
    modelName = "plant_regression_v0.4"
    batchSize = 32
    swin_lr = 1e-4
    mlp_max_lr = 1e-3
    mlp_min_lr = 1e-6
    warm_up_epoch = 5
    restart = 100
    train_epoch = 100
    total_epoch = train_epoch + warm_up_epoch
    weight_decay = 1e-2
    img_size = 224
    pretrained_model = "swin_base_patch4_window7_224_22k.pth"
    img_transform = {
        "train": A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomSizedCrop(
                    (448, 512),
                    (img_size, img_size),
                    w2h_ratio=1.0,
                    p=0.75,
                ),
                A.Resize(img_size, img_size),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.25
                ),
                A.ImageCompression(quality_lower=85, quality_upper=100, p=0.25),
                A.ToFloat(),
                A.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], max_pixel_value=1
                ),
                ToTensorV2(),
            ]
        ),
        "val": A.Compose(
            [
                A.Resize(img_size, img_size),
                A.ToFloat(),
                A.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], max_pixel_value=1
                ),
                ToTensorV2(),
            ]
        ),
    }
    feature_scaler = StandardScaler()
    # 实例化训练数据集
    train_dataset = MyDataSet(
        root_dir=data_root,
        img_transform=img_transform["train"],
        feature_scaler=feature_scaler,
        is_train=True,
    )

    # 实例化验证数据集
    val_dataset = MyDataSet(
        root_dir=data_root,
        img_transform=img_transform["val"],
        feature_scaler=feature_scaler,
        is_train=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchSize,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchSize,
        shuffle=False,
        pin_memory=True,
    )

    iterations = total_epoch * len(train_loader)
    print("iterarions:{}".format(iterations))
    image_encoder = swin_transformer().to(device)
    number_encoder = linear_encoder().to(device)
    decoder = linear_decoder(in_channel=1350, out_channel=6, hidden_channel=256).to(
        device
    )
    # define R-Square loss function
    r2metric = RSquare(is_loss=False).to(device)
    loss_func = RSquare(is_loss=True).to(device)
    lastEpoch = 0
    lossArray = []
    image_optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, image_encoder.parameters()),
        lr=swin_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    other_optimizer = optim.AdamW(
        [{"params": number_encoder.parameters()}, {"params": decoder.parameters()}],
        lr=mlp_max_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
    )
    # warm up lr
    alpha = mlp_min_lr / mlp_max_lr

    def lr_func(epoch):
        if epoch < warm_up_epoch:
            return (epoch + 1) / warm_up_epoch
        else:
            return alpha + 0.5 * (1 - alpha) * (
                1 + np.cos((epoch + 1 - warm_up_epoch) / restart * np.pi)
            )

    scheduler1 = optim.lr_scheduler.LambdaLR(image_optimizer, lr_lambda=lr_func)
    scheduler2 = optim.lr_scheduler.LambdaLR(other_optimizer, lr_lambda=lr_func)

    weightDir = os.path.join("checkPoints", modelName)
    logDir = os.path.join(weightDir, "logs")
    if os.path.exists(weightDir):
        weightsList = [
            x
            for x in os.listdir(weightDir)
            if os.path.isfile(os.path.join(weightDir, x)) and x.endswith("tar")
        ]
        if len(weightsList):
            weightsList.sort(key=lambda x: os.path.getctime(os.path.join(weightDir, x)))
            fileName = weightsList[-1]
            filePath = os.path.join(weightDir, fileName)
            chkPoint = torch.load(filePath)
            print("load weightFile:{}".format(filePath))
            if torch.cuda.device_count() > 1:
                image_encoder.module.load_state_dict(
                    chkPoint["image_encoder_state_dict"]
                )
            else:
                image_encoder.load_state_dict(chkPoint["image_encoder_state_dict"])
            number_encoder.load_state_dict(chkPoint["num_encoder_state_dict"])
            decoder.load_state_dict(chkPoint["decoder_state_dict"])
            image_optimizer.load_state_dict(chkPoint["image_optimizer_state_dict"])
            other_optimizer.load_state_dict(chkPoint["other_optimizer_state_dict"])
            scheduler1.load_state_dict(chkPoint["scheduler1_state_dict"])
            scheduler2.load_state_dict(chkPoint["scheduler2_state_dict"])
            loss = chkPoint["loss"]
            lastEpoch = chkPoint["epoch"]
    else:
        if pretrained_model:
            pretrained_filePath = os.path.join("pretrained", pretrained_model)
            weights_dict = torch.load(pretrained_filePath, map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(image_encoder.load_state_dict(weights_dict, strict=False))

        os.makedirs(weightDir)

    # add logger
    logFile = os.path.join(weightDir, "train.log")
    logger = getDualOutputLogger(logFile)

    # add tensorboard writer
    writer = SummaryWriter(logDir)

    image_encoder.train()
    number_encoder.train()
    decoder.train()
    for epoch in range(lastEpoch + 1, lastEpoch + total_epoch + 1):
        print(
            "start epoch {}: ,lr:{}".format(
                epoch, other_optimizer.state_dict()["param_groups"][0]["lr"]
            )
        )
        lossSum = 0.0
        for index, (img, aux, label) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            img = img.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            img_feature = image_encoder(img)
            num_feature = number_encoder(aux)
            out = decoder(torch.cat((img_feature, num_feature), dim=1))
            loss = loss_func(out, label)
            lossSum += loss.item()
            image_optimizer.zero_grad()
            other_optimizer.zero_grad()
            loss.backward()
            image_optimizer.step()
            other_optimizer.step()

        loss_mean = lossSum / len(train_loader)
        lossArray.append(loss_mean)
        if len(lossArray) > 10:
            lossArray.pop(0)

        logger.info(
            "end epoch{} train,lr:{},loss_total:{:.4f},mean:{:.4f},std:{:.4f}".format(
                epoch,
                image_optimizer.state_dict()["param_groups"][0]["lr"],
                loss_mean,
                np.mean(lossArray),
                np.std(lossArray),
            )
        )
        print("last 10 loss:", end="")
        for i in lossArray:
            print(" {:.4f}".format(i), end="")
        print("")
        scheduler1.step()
        scheduler2.step()
        if not epoch % 2:
            image_encoder.eval()
            number_encoder.eval()
            decoder.eval()
            vali_loss = validate(
                image_encoder,
                number_encoder,
                decoder,
                val_loader,
                device,
                epoch,
                logger,
            )
            image_encoder.train()
            number_encoder.train()
            decoder.train()
            writer.add_scalars(
                "Loss",
                (
                    {
                        "loss_total": loss_mean,
                        "vali_loss": vali_loss,
                    }
                ),
                epoch,
            )
            saveCheckPoint(
                epoch,
                image_encoder,
                number_encoder,
                decoder,
                image_optimizer,
                other_optimizer,
                scheduler1,
                scheduler2,
                loss,
                weightDir,
            )
        else:
            writer.add_scalars(
                "Loss",
                (
                    {
                        "loss_total": loss_mean,
                    }
                ),
                epoch,
            )
    writer.close()


if __name__ == "__main__":
    train()
