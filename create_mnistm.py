import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
from utils import utils

# 返回融合后[mnist_width,mnist_height]的图片
def transpose_mnist_bdbs(mnist_img,bdbs_img):
    mnist_width,mnist_height = mnist_img.shape
    bdbs_width,bdbs_height = bdbs_img.shape
    x = np.random.randint(0,bdbs_width-mnist_width)
    y = np.random.randint(0,bdbs_height-mnist_height)
    cropped_bdbs_img = bdbs_img[x:x+mnist_width,y:y+mnist_height]
    return np.abs(mnist_img-cropped_bdbs_img).astype(np.uint8)

# mnist_imgs:[num_mnist,width,height], mnist_labels:[batchsize], bdbs_imgs:长度num_bdbs的list[bdbs_img]
# 返回mnistm_imgs:ndarray[60000,28,28],mnistm_labels:ndarray[60000]
def create_mnistm(mnist_imgs,mnist_labels,bdbs_imgs):
    mnistm_imgs = []
    for i in range(len(mnist_imgs)):
        mnist_img = mnist_imgs[i]
        bdbs_img = random.choice(bdbs_imgs)
        mnistm_img = transpose_mnist_bdbs(mnist_img,bdbs_img)
        mnistm_imgs.append(mnistm_img)
    mnistm_imgs = np.array(mnistm_imgs)
    return mnistm_imgs,mnist_labels


dataset_dir = 'dataset'
bdbs_imgs_dir_path = 'dataset/BDBS500/images/train/data'
#[60000,28,28]
mnist_train_imgs = utils.parse_mnist('dataset/MNIST/raw/train-images-idx3-ubyte.gz')
mnist_train_labels = utils.parse_mnist('dataset/MNIST/raw/train-labels-idx1-ubyte.gz')
mnist_test_imgs = utils.parse_mnist('dataset/MNIST/raw/t10k-images-idx3-ubyte.gz')
mnist_test_labels = utils.parse_mnist('dataset/MNIST/raw/t10k-labels-idx1-ubyte.gz')
bdbs_imgs = []
bdbs_imgs_names_list = os.listdir(bdbs_imgs_dir_path)
for bdbs_img_name in bdbs_imgs_names_list:
    bdbs_img_path = os.path.join(bdbs_imgs_dir_path,bdbs_img_name)
    bdbs_img = cv2.imread(bdbs_img_path,0)
    bdbs_imgs.append(bdbs_img)
mnistm_train_imgs,mnistm_train_labels = create_mnistm(mnist_train_imgs,mnist_train_labels,bdbs_imgs)
mnistm_train_imgs = np.expand_dims(mnistm_train_imgs,axis=-1) #ndarray[60000,28,28,1]
mnistm_train_imgs = np.concatenate([mnistm_train_imgs,mnistm_train_imgs,mnistm_train_imgs],axis=3) #ndarray[60000,28,28,3]
mnistm_test_imgs,mnistm_test_labels = create_mnistm(mnist_test_imgs,mnist_test_labels,bdbs_imgs)
mnistm_test_imgs = np.expand_dims(mnistm_test_imgs,axis=-1) #ndarray[10000,28,28,1]
mnistm_test_imgs = np.concatenate([mnistm_test_imgs,mnistm_test_imgs,mnistm_test_imgs],axis=3) #ndarray[10000,28,28,3]
with open(os.path.join(dataset_dir, 'mnistm_data.pkl'), 'wb') as f:
    pkl.dump({'train_imgs': mnistm_train_imgs,
              'train_labels': mnistm_train_labels,
              'validate_imgs': mnistm_test_imgs,
              'validate_labels': mnistm_test_labels}, f, pkl.HIGHEST_PROTOCOL)
