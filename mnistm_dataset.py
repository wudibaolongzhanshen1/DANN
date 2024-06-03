import torch
from torch.utils.data import Dataset


class MnistmDataset(Dataset):
    def __init__(self, mnistm_imgs, mnistm_labels, transform=None):
        self.mnistm_imgs = mnistm_imgs
        self.mnistm_labels = mnistm_labels
        self.transform = transform

    def __len__(self):
        return len(self.mnistm_imgs)

    def __getitem__(self, idx):
        img = self.mnistm_imgs[idx]
        label = self.mnistm_labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # dataset = ((0.56998216, 0.72663738, 0.3706266),(0.3403586 , 0.13931333, 0.71030221)) = (imgarray,label1,label2)。为tuple(tulple,tulple)
        # batch为zip(list(tuple(tensor_array(3*224*224),label1,label2,...))) = [([img1[3*224*224],i1_label1,i1_label2), ([img2[3*224*224],i2_label1,i2_label2), ([img3[3*224*224],i3_label1,i3_label2),...]
        # batch = zip(list[tuple(img1,i1_label1,i1_label2),tuple(img2,i2_label1,i2_label2),...])
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # zip()可以理解为压缩，zip(*)可以理解为解压,但是解压后还是zip类型，需要转变为tuple
        # 注意虽然zip(*)可以理解为解压缩，但是解压缩后的数据排列方式发生了变化：zip((i1,l1),(i2,l2)) = ((i1,i2),(l1,l2))
        # tuple(zip(*batch)) = tuple(tuple(img1,img2,img3,...),tuple(i1_label1,i2_label1,i3_label1,...),tuple(i1_label2,i2_label2,...)) = ((img1,img2,img3,...),(i1_label1,i2_label1,i3_label1,...),(i1_label2,i2_label2,...))
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
