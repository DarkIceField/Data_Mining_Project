import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.loss import RSquare
from utils.my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as swin_transformer
from model import linear_encoder
from utils.utils import read_split_data, train_one_epoch, evaluate
from tqdm import tqdm
from typing import List
from torch import nn
import logging
import numpy as np
from torch.cuda.amp import autocast, GradScaler


def saveCheckPoint(
    epoch: int,
    image_encoder: nn.Module,
    num_encoder: nn.Module,
    decoder: nn.Module,
    image_optimizer: optim.Optimizer,
    other_optimizer: optim.Optimizer,
    scaler: GradScaler,
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
            "scaler_state_dict": scaler.state_dict(),
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
    modelName = "plant_regression_v0.1"
    batchSize = 32
    epochs = 300
    swin_lr = 1e-5
    mlp_lr = 1e-2
    img_size = 224
    pretrained_model = "swin_tiny_patch4_window7_224.pth"
    img_transform = {
        "train": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomResizedCrop((img_size, img_size),scale=(0.6,1),antialias=True),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(0.1, 0.1, (0.45, 0.55), 0.1),
                        transforms.RandomErasing(p=1, scale=(0.06, 0.15)),
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.RandomVerticalFlip(p=1),
                        transforms.RandomRotation((3.6, 18)),
                    ],
                    p=0.5,
                ),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # 实例化训练数据集
    train_dataset = MyDataSet(
        csv_path=os.path.join(data_root, "train", "train.csv"),
        img_transform=img_transform["train"],
    )

    # 实例化验证数据集
    val_dataset = MyDataSet(
        csv_path=os.path.join(data_root, "validate", "val.csv"),
        img_transform=img_transform["val"],
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

    iterations = epochs * len(train_loader)
    print("iterarions:{}".format(iterations))
    image_encoder = swin_transformer().to(device)
    number_encoder = linear_encoder().to(device)
    decoder = nn.Linear(1094, 6).to(device)
    # define R-Square loss function
    r2loss = RSquare(is_loss=True).to(device)
    lastEpoch = 0
    lossArray = []
    image_optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, image_encoder.parameters()),
        lr=swin_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    other_optimizer = optim.AdamW(
        [{"params": number_encoder.parameters()}, {"params": decoder.parameters()}],
        lr=mlp_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )
    #define scaler
    scaler = GradScaler()

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
            number_encoder.load_state_dict(chkPoint["number_encoder_state_dict"])
            decoder.load_state_dict(chkPoint["decoder_state_dict"])
            image_optimizer.load_state_dict(chkPoint["image_optimizer_state_dict"])
            other_optimizer.load_state_dict(chkPoint["other_optimizer_state_dict"])
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
    for epoch in range(lastEpoch + 1, lastEpoch + epochs + 1):
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
            with autocast():
                img_feature = image_encoder(img)
                num_feature = number_encoder(aux)
                out = decoder(torch.cat((img_feature, num_feature), dim=1))
                loss = r2loss(out, label)
            lossSum += loss.item()
            image_optimizer.zero_grad()
            other_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(image_optimizer)
            scaler.step(other_optimizer)
            scaler.update()

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
        if not epoch % 5:
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
                scaler,
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
