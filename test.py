import os
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import torch
from torchvision import transforms
import torchvision.transforms.functional as transF
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
from utils.mDataset import *
from utils.load import load_pretrained
from utils.evaluator import Evaluator
import matplotlib.pyplot as plt
from utils.visualize import drawLineChart, writeMetricsRow
import cv2 as cv
from tqdm import tqdm
from utils.loss import CompetitiveMoELoss

# from utils.lossL1 import ir_loss, vi_loss, ssim_loss, gra_loss
import logging
from models.model_convnext import ICMFusion
from torchvision import models
from typing import Optional, List
import shutil
import pandas as pd
from utils.fixSeed import *


def validate(
    gate_model: nn.Module,
    model: nn.Module,
    dataSets: List[ImgPairSubSet],
    device: str,
    weight: List[int],
    savePath: str,
    epoch: int,
    metricNames: List[str],
    lossType: str,
    stage: int,
    logger: logging.Logger,
):
    cptLoss = CompetitiveMoELoss(weights=weight, lossType=lossType, stage=stage)
    lossNames = (
        ["loss_validate", "loss_ir", "loss_vi", "loss_ssim", "loss_gra"]
        if stage == 1
        else ["loss_validate", "loss_int", "loss_ssim", "loss_gra"]
    )
    losses = np.empty((5, 0)) if stage == 1 else np.empty((4, 0))
    metrics_allSet_str = ""
    separatorStr = "=" * 100
    print("start test:")
    for test_set in dataSets:
        metrics_array = np.empty((len(metricNames), 0))
        dirPath = os.path.join(savePath, "current_results", test_set.getName())
        if os.path.exists(dirPath):
            shutil.rmtree(dirPath)
        os.makedirs(dirPath)

        with torch.no_grad():
            for index, (vi, ir, imgName) in tqdm(
                enumerate(test_set), total=len(test_set)
            ):
                ir, vi = ir.unsqueeze_(0), vi.unsqueeze_(0)
                ir = ir.to(device)
                vi = vi.to(device)
                illu_factor = gate_model(vi)
                illu_factor = torch.sigmoid(illu_factor)
                vi = transF.rgb_to_grayscale(vi)
                ndIR = tensorToNdImg(ir)
                ndVI = tensorToNdImg(vi)
                out1, out2 = model(vi, ir)
                res = illu_factor * out1 + (1 - illu_factor) * out2
                ndRes = tensorToNdImg(res)
                uintRes = ndRes.astype(np.uint8)
                cv.imwrite(
                    os.path.join(dirPath, "{}_epoch{}.png".format(imgName, epoch)),
                    uintRes,
                )
                loss, lossList = cptLoss(vi, ir, out1, out2, illu_factor)
                losses = np.column_stack((losses, lossList))

                metricList = getMetrics(ndRes, ndIR, ndVI, metricNames)
                metrics_array = np.column_stack((metrics_array, metricList))

        metrics_mean = np.mean(metrics_array, axis=1)
        model_name = os.path.basename(savePath)
        writeMetricsRow(
            os.path.join(savePath, model_name + "_validate_metrics.xlsx"),
            test_set.getName(),
            metricNames,
            metrics_mean,
            str(epoch),
        )
        metricStr = "metrics on {}: ".format(test_set.getName())
        for i in range(len(metricNames)):
            metricStr += "{}:{:.4f},".format(metricNames[i], metrics_mean[i])
        metrics_allSet_str += separatorStr + "\n" + metricStr + "\n"

    loss_mean = np.mean(losses, axis=1)
    lossStr = "loss: "
    for i in range(len(lossNames)):
        lossStr += "{}:{:.4f},".format(lossNames[i], loss_mean[i])

    logger.info("end epoch{} test,\n{},\n{}".format(epoch, lossStr, metrics_allSet_str))

    return loss_mean[0]


def tensorToNdImg(tensor):
    ndRes = tensor.squeeze()
    ndRes = (
        (ndRes - torch.min(ndRes)) / (torch.max(ndRes) - torch.min(ndRes) + 1e-8) * 255
    )
    ndRes = ndRes.detach().cpu().numpy()
    return ndRes


def getMetrics(ndRes, ndIR, ndVI, metricNames):
    single_source_metrics = ["EN", "SD", "SF", "AG"]
    metricList = []
    for metric in metricNames:
        metricFunc = getattr(Evaluator(), metric)
        if metric in single_source_metrics:
            metricList.append(metricFunc(ndRes))
        else:
            metricList.append(metricFunc(ndRes, ndIR, ndVI))
    return metricList


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("using {} device.".format(device))

    # load pretrained model
    # IFCNNmodel = IFCNN().to(device)
    # weightFile = os.path.join('checkPoints'+'IFCNN','IFCNN-MAX.pth')
    # IFCNNmodel.load_state_dict(torch.load(weightFile))
    # IFCNNmodel.eval()
    gate_model = models.resnet18()
    gate_model.fc = nn.Linear(512, 1)
    gate_model.to(device)
    mymodel = ICMFusion(patch_size=1).to(device)
    if torch.cuda.device_count() > 1:
        print("use {} GPUS".format(torch.cuda.device_count()))
        mymodel = nn.DataParallel(mymodel)

    savemodelName = "model_v0.4.7"
    weightFile = os.path.join("pretrained", savemodelName + ".tar")
    chkPoint = torch.load(weightFile)
    print("load weightFile:{}".format(weightFile))
    gate_model.load_state_dict(chkPoint["gate_model_state_dict"])
    if torch.cuda.device_count() > 1:
        mymodel.module.load_state_dict(chkPoint["fuse_model_state_dict"])
    else:
        mymodel.load_state_dict(chkPoint["fuse_model_state_dict"])
    gate_model.eval()
    mymodel.eval()

    totalParam = sum([param.nelement() for param in mymodel.parameters()])
    print("Total parameters number of model: {:.4f}M".format(totalParam / 1000000))

    datasets = [
        "MSRS",
        "RoadScene",
        "M3FD",
    ]  # Infrared-Visual image datasets
    is_save = True  # if you do not want to save images, then change its value to False
    is_show = False
    is_evalute = False
    max_image_num = None
    testRoot = "../../fusion_datasets/test/"
    # saveRoot = os.path.join("results", savemodelName)
    saveRoot = "../../fusion_results/ICMFusion"
    metrics_caculated = ["EN", "SD", "SF", "MI", "SCD", "VIFF", "Qabf", "SSIM"]
    metrics_all_str = ""

    for j in range(len(datasets)):
        print(datasets[j])
        setRoot = os.path.join(testRoot, datasets[j])
        test_dataset = ImgPairSubSet(
            img_dir=setRoot, vi_open_mode="RGB", maxNum=max_image_num
        )
        metrics_sum = np.zeros(len(metrics_caculated))
        dirPath = os.path.join(saveRoot, datasets[j])
        if not os.path.exists(dirPath):
            os.makedirs(dirPath, exist_ok=True)
        begin_time = time.time()
        for vi, ir, img_name in test_dataset:
            vi, ir = vi.unsqueeze_(0), ir.unsqueeze_(0)
            vi = vi.to(device)
            ir = ir.to(device)
            print(img_name)
            # perform image fusion
            with torch.no_grad():
                illu_factor = gate_model(vi)
                illu_factor = torch.sigmoid(illu_factor)
                vi = transF.rgb_to_grayscale(vi)
                res1, res2 = mymodel(vi, ir)
                res = illu_factor * res1 + (1 - illu_factor) * res2
                ndRes = tensorToNdImg(res)
                ndIR = tensorToNdImg(ir)
                ndVI = tensorToNdImg(vi)
                if is_evalute:
                    metrics = getMetrics(ndRes, ndIR, ndVI, metrics_caculated)
                    metrics_sum += np.array(metrics)
                uintRes = ndRes.astype(np.uint8)
                if is_show:
                    cv.imshow("a", uintRes)
                    cv.waitKey(20000)
                # save fused images
                if is_save:
                    cv.imwrite(os.path.join(dirPath, img_name + ".png"), uintRes)

        mean_proc_time = (time.time() - begin_time) / len(test_dataset)
        print(
            "Mean processing time of {} dataset: {:.3}s".format(
                datasets[j], mean_proc_time
            )
        )
        if is_evalute:
            metrics_mean = metrics_sum / len(test_dataset)
            writeMetricsRow(
                os.path.join(saveRoot, savemodelName + "metrics.xlsx"),
                datasets[j],
                metrics_caculated,
                metrics_mean,
                epoch="Ours",
            )
            metricStr = "metrics on {} dataset: ".format(datasets[j])
            for i in range(len(metrics_caculated)):
                metricStr += "{}:{:.4f},".format(metrics_caculated[i], metrics_mean[i])
            metrics_all_str += metricStr + "\n"

    if metrics_all_str:
        print(metrics_all_str)


if __name__ == "__main__":
    test()
