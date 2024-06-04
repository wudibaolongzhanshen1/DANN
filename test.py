import pickle
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import logging
import model
from utils import utils
from mnistm_dataset import MnistmDataset

device = torch.device("cuda")
config_path = 'config/2024-06-03-22-23-47/config.txt'
config_dict = {}
with open(config_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            i = line.find(':')
            key = line[:i]
            value = line[i + 1:]
            config_dict[key] = value
logging.basicConfig(level=logging.DEBUG,
                    filename=config_dict['logs_dir'].strip() + '/train.log',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
batch_size = int(config_dict['batch_size'])
learning_rate = float(config_dict['init_learning_rate'])
epoch = int(config_dict['epoch'])
mnist_train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((28, 28))]
)
mnist_test_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((28, 28))]
)

mnistm_train_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((28, 28)),
     transforms.Normalize(mean=utils.str2array(config_dict['pixel_mean']),
                          std=utils.str2array(config_dict['pixel_std']))]
)
mnist_train_dataset = MNIST(root='dataset', train=True, download=True, transform=mnist_train_transforms)
mnist_test_dataset = MNIST(root='dataset', train=False, download=True, transform=mnist_test_transforms)
mnist_train_loader = DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
mnist_test_loader = DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
mnist_train_num = len(mnist_train_dataset)
mnist_test_num = len(mnist_test_dataset)

# with open('dataset\\mnist_data.pkl', 'rb') as f:
#     a = pickle.load(f)
#     mnist_train_imgs = a['train_imgs']  # (60000, 28, 28, 3)
#     mnist_test_imgs = a['validate_imgs']  # (10000, 28, 28, 3)
#     mnist_train_labels = a['train_labels']  # (60000,)
#     mnist_test_labels = a['validate_labels']  # (10000,)
#     mnist_train_dataset = MnistmDataset(mnist_train_imgs, mnist_train_labels, transform=mnist_train_transforms)
#     mnist_test_dataset = MnistmDataset(mnist_test_imgs, mnist_test_labels, transform=mnist_test_transforms)
#     mnist_train_loader = DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)
#     mnist_test_loader = DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=True)
#     mnist_train_num = len(mnist_train_dataset)
#     mnist_test_num = len(mnist_test_dataset)

with open('dataset\\mnistm_data.pkl', 'rb') as f:
    a = pickle.load(f)
    mnistm_train_imgs = a['train_imgs']  # (60000, 28, 28, 3)
    mnistm_test_imgs = a['validate_imgs']  # (10000, 28, 28, 3)
    mnistm_train_labels = a['train_labels']  # (60000,)
    mnistm_test_labels = a['validate_labels']  # (10000,)
    mnistm_train_dataset = MnistmDataset(mnistm_train_imgs, mnistm_train_labels, transform=mnistm_train_transforms)
    mnistm_test_dataset = MnistmDataset(mnistm_test_imgs, mnistm_test_labels, transform=mnistm_train_transforms)
    mnistm_train_loader = DataLoader(dataset=mnistm_train_dataset, batch_size=batch_size, shuffle=True)
    mnistm_test_loader = DataLoader(dataset=mnistm_test_dataset, batch_size=batch_size, shuffle=True)
    mnistm_train_num = len(mnistm_train_dataset)
    mnistm_test_num = len(mnistm_test_dataset)

SourceDloss = nn.CrossEntropyLoss().to(device)
TargetDloss = nn.CrossEntropyLoss().to(device)
ClsLoss = nn.CrossEntropyLoss().to(device)
net = model.DANN().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
for step,(minst_imgs, mnist_labels) in enumerate(mnist_train_loader):
    print(minst_imgs == 0)
    print(mnist_labels)
    break