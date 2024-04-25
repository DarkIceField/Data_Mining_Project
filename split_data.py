import os
from shutil import copy, rmtree
import random
from tqdm import tqdm
import pandas as pd


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    # 指向你解压后的flower_photos文件夹
    data_root = os.path.join(os.getcwd(), "data", "train&val")
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
    tabular_df = pd.read_csv(tabular_path)
    train_df = pd.DataFrame(columns=tabular_df.columns)
    train_df["path"] = []
    train_df.to_csv(train_csv_path, header=True, index=False)
    val_df = pd.DataFrame(columns=tabular_df.columns)
    val_df["path"] = []
    val_df.to_csv(val_csv_path, header=True, index=False)
    num = len(images)
    # 随机采样验证集的索引
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in tqdm(enumerate(images), total=len(images)):
        raw_img_name, _ = os.path.splitext(image)
        if image in eval_index:
            # 将分配至验证集中的文件复制到相应目录
            image_path = os.path.join(img_dir, image)
            copy(image_path, val_img_dir)
            sgl_row_df = tabular_df[tabular_df["id"] == int(raw_img_name)].copy()
            sgl_row_df["path"] = [os.path.join(val_img_dir, image)]
            sgl_row_df.to_csv(val_csv_path, mode="a", header=False, index=False)
        else:
            # 将分配至训练集中的文件复制到相应目录
            image_path = os.path.join(img_dir, image)
            copy(image_path, train_img_dir)
            sgl_row_df = tabular_df[tabular_df["id"] == int(raw_img_name)].copy()
            sgl_row_df["path"] = [os.path.join(train_img_dir, image)]
            sgl_row_df.to_csv(train_csv_path, mode="a", header=False, index=False)

    print("processing done!")


if __name__ == "__main__":
    main()
