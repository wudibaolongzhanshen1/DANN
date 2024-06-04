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
     transforms.Resize((28, 28))]
)
# mnist_train_dataset = MNIST(root='dataset', train=True, download=True, transform=mnist_train_transforms)
# mnist_test_dataset = MNIST(root='dataset', train=False, download=True, transform=mnist_test_transforms)
# mnist_train_loader = DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)
# mnist_test_loader = DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=True)
# mnist_train_num = len(mnist_train_dataset)
# mnist_test_num = len(mnist_test_dataset)
#
with open('dataset\\mnist_data.pkl', 'rb') as f:
    a = pickle.load(f)
    mnist_train_imgs = a['train_imgs']  # (60000, 28, 28, 3)
    mnist_test_imgs = a['validate_imgs']  # (10000, 28, 28, 3)
    mnist_train_labels = a['train_labels']  # (60000,)
    mnist_test_labels = a['validate_labels']  # (10000,)
    mnist_train_dataset = MnistmDataset(mnist_train_imgs, mnist_train_labels, transform=mnist_train_transforms)
    mnist_test_dataset = MnistmDataset(mnist_test_imgs, mnist_test_labels, transform=mnist_test_transforms)
    mnist_train_loader = DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=True)
    mnist_train_num = len(mnist_train_dataset)
    mnist_test_num = len(mnist_test_dataset)

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
# 训练,0代表原域，1代表目标域
for i in range(epoch):
    net.train()
    len_dataloader = min(len(mnist_train_loader), len(mnistm_train_loader))
    datasource_iter = iter(mnist_train_loader)
    datatarget_iter = iter(mnistm_train_loader)
    for step in range(len_dataloader):
        mnist_imgs, mnist_labels = next(datasource_iter)
        mnistm_imgs, mnistm_labels = next(datatarget_iter)
        mnist_imgs = mnist_imgs.to(device)
        mnist_labels = mnist_labels.to(device)
        mnistm_imgs = mnistm_imgs.to(device)
        mnistm_labels = mnistm_labels.to(device)
        # print(mnist_imgs.min(),' ',mnist_imgs.max())
        # print(mnistm_imgs.min(),' ',mnistm_imgs.max())
        # p为当前训练的进度
        p = float(i * len_dataloader + step) / epoch / len_dataloader
        lambda_ = 2. / (1. + np.exp(-10. * p)) - 1
        source_cls_pred_results, source_domain_pred_results = net(mnist_imgs, lambda_)
        target_cls_pred_results, target_domain_pred_results = net(mnistm_imgs, lambda_)
        cls_loss = ClsLoss(source_cls_pred_results, mnist_labels)
        # 第二个参数一定要是long类型的，这是torch库函数规定的
        source_domain_loss = SourceDloss(source_domain_pred_results,
                                         torch.zeros(source_domain_pred_results.shape[0]).long().to(device))
        target_domain_loss = TargetDloss(target_domain_pred_results,
                                         torch.ones(target_domain_pred_results.shape[0]).long().to(device))
        loss = cls_loss + source_domain_loss + target_domain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                         % (i, step + 1, len_dataloader, cls_loss.data.cpu().numpy(),
                            source_domain_loss.data.cpu().numpy(), target_domain_loss.data.cpu().item()))
        sys.stdout.flush()

    net.eval()
    mnistm_cls_accuracy = 0.
    mnistm_domain_accuracy = 0.
    # 重置迭代器
    datatarget_iter = iter(mnistm_train_loader)
    for step in range(len(mnistm_train_loader)):
        mnistm_imgs, mnistm_labels = next(datatarget_iter)
        mnistm_imgs = mnistm_imgs.to(device)
        mnistm_labels = mnistm_labels.to(device)
        class_pred, domain_pred = net(mnistm_imgs)
        sm = nn.Softmax()
        class_pred = sm(class_pred)
        class_pred = torch.argmax(class_pred, dim=1)
        domain_pred = sm(domain_pred)
        domain_pred = torch.argmax(domain_pred, dim=1)
        mnistm_cls_accuracy += torch.sum(class_pred == mnistm_labels)
        mnistm_domain_accuracy += torch.sum(domain_pred == torch.ones_like(domain_pred))
    mnistm_cls_accuracy = mnistm_cls_accuracy / len(mnistm_train_loader)
    mnistm_domain_accuracy = mnistm_domain_accuracy / len(mnistm_train_loader)

    best_mnist_cls_accuracy = 0.
    mnist_cls_accuracy = 0.
    mnist_domain_accuracy = 0.
    datasource_iter = iter(mnist_train_loader)
    for step in range(len(mnist_test_loader)):
        mnist_imgs, mnist_labels = next(datasource_iter)
        mnist_imgs = mnist_imgs.to(device)
        mnist_labels = mnist_labels.to(device)
        class_pred, domain_pred = net(mnist_imgs)
        sm = nn.Softmax()
        class_pred = torch.argmax(sm(class_pred), dim=1)
        domain_pred = torch.argmax(sm(domain_pred), dim=1)
        mnistm_cls_accuracy += torch.sum(class_pred == mnist_labels)
        mnistm_domain_accuracy += torch.sum(domain_pred == torch.ones_like(domain_pred))
    mnist_cls_accuracy = mnist_cls_accuracy / len(mnist_train_loader)
    mnist_domain_accuracy = mnist_domain_accuracy / len(mnist_train_loader)
    if mnist_cls_accuracy > best_mnist_cls_accuracy:
        best_mnist_cls_accuracy = mnist_cls_accuracy
        torch.save(net.state_dict(), 'model/best_model.pth')
    print(
        '\r [epoch: %d / all %d], mnistm_cls_accuracy: %f, mnistm_domain_accuracy: %f, mnist_cls_accuracy: %f, mnist_domain_accuracy: %f' \
        % (i, epoch, mnistm_cls_accuracy, mnistm_domain_accuracy, mnist_cls_accuracy, mnist_domain_accuracy))
    # use logging
    logging.info(
        '\r [epoch: %d / all %d], mnistm_cls_accuracy: %f, mnistm_domain_accuracy: %f, mnist_cls_accuracy: %f, mnist_domain_accuracy: %f' \
        % (i, epoch, mnistm_cls_accuracy, mnistm_domain_accuracy, mnist_cls_accuracy, mnist_domain_accuracy))
