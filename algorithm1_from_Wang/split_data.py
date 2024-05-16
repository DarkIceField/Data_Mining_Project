import os
from shutil import copy, rmtree
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv

tabular_df = pd.read_csv(r'C:\data\plant\train.csv')
test_df = pd.read_csv(r'C:\data\plant\test.csv')
tabular_df.head()
mean_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']

for column in mean_columns:
    upper_quantile = tabular_df[column].quantile(0.98)
    lower_quantile = tabular_df[column].quantile(0.02)
    tabular_df = tabular_df[(tabular_df[column] < upper_quantile)]
    tabular_df = tabular_df[(tabular_df[column] > lower_quantile)]


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def split_data():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    data_root = r'C:\data\plant'
    img_dir = os.path.join(data_root, "train_images")
    tabular_path = os.path.join(data_root, "train.csv")
    assert os.path.exists(img_dir), "path '{}' does not exist.".format(img_dir)

    # 建立保存训练集的文件夹
    train_root = os.path.join(os.getcwd(), "data", "train")
    mk_file(train_root)
    train_img_dir = os.path.join(train_root, "train_images")
    mk_file(train_img_dir)
    train_csv_path = os.path.join(train_root, "train.csv")

    # 建立保存验证集的文件夹
    val_root = os.path.join(os.getcwd(), "data", "validate")
    mk_file(val_root)
    val_img_dir = os.path.join(val_root, "val_images")
    mk_file(val_img_dir)
    val_csv_path = os.path.join(val_root, "val.csv")

    images = os.listdir(img_dir)
    train_df = pd.DataFrame(columns=tabular_df.columns)
    train_df["path"] = []
    train_df.to_csv(train_csv_path, header=True, index=False)
    val_df = pd.DataFrame(columns=tabular_df.columns)
    val_df["path"] = []
    val_df.to_csv(val_csv_path, header=True, index=False)
    num = len(tabular_df)
    # 随机采样验证集的索引
    imgid_list = tabular_df['id'].tolist()
    eval_index = random.sample(imgid_list, k=int(num * split_rate))
    images = os.listdir(img_dir)
    for index, img_id in tqdm(enumerate(imgid_list), total=len(imgid_list)):
        img_name = str(img_id) + ".jpeg"
        image_path = os.path.join(img_dir, img_name)
        if img_id in eval_index:
            # 将分配至验证集中的文件复制到相应目录
            copy(image_path, val_img_dir)
            sgl_row_df = tabular_df[tabular_df["id"] == int(img_id)].copy()
            sgl_row_df["path"] = [os.path.join(val_img_dir, img_name)]
            sgl_row_df.to_csv(val_csv_path, mode="a", header=False, index=False)
        else:
            # 将分配至训练集中的文件复制到相应目录
            copy(image_path, train_img_dir)
            sgl_row_df = tabular_df[tabular_df["id"] == int(img_id)].copy()
            sgl_row_df["path"] = [os.path.join(train_img_dir, img_name)]
            sgl_row_df.to_csv(train_csv_path, mode="a", header=False, index=False)

    print("processing done!")
# split_data()

test_img_dir = r'C:\data\plant\test_images'
test_images = os.listdir(test_img_dir)
for image in test_images:
    raw_img_name, _ = os.path.splitext(image)
    imgPath = os.path.join(test_img_dir, image)
    test_df.loc[test_df["id"] == int(raw_img_name),"path"] = imgPath
test_df.to_csv("data/test/test.csv", mode="w", header=True, index=False)