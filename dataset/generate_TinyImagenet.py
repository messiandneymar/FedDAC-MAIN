import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file
from torchvision.datasets import ImageFolder, DatasetFolder
import requests
import zipfile

random.seed(1)
np.random.seed(1)
num_clients = 40
dir_path = "TinyImagenet/"


# 自定义的 ImageFolder 类
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


# 下载并解压数据集的函数
def download_and_extract(url, download_dir):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    filename = os.path.join(download_dir, "tiny-imagenet-200.zip")

    # 下载文件
    print("开始下载数据集...")
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print("下载完成！")

    # 解压文件
    print("正在解压数据集...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    print("解压完成！")


# 分配数据给用户
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # 设置训练/测试数据目录
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # 获取数据
    raw_data_path = f'{dir_path}/rawdata/'
    if not os.path.exists(raw_data_path):
        # 这里使用了 Python 自带的下载和解压方式
        download_and_extract('http://cs231n.stanford.edu/tiny-imagenet-200.zip', raw_data_path)
    else:
        print('rawdata already exists.\n')

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = ImageFolder_custom(root=dir_path + 'rawdata/tiny-imagenet-200/train/', transform=transform)
    testset = ImageFolder_custom(root=dir_path + 'rawdata/tiny-imagenet-200/val/', transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # 数据分配给客户端
    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=20)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)
