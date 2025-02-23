import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
import math
from os.path import isfile, join
from augment import *

warnings.filterwarnings('ignore')

mapping_dvscifar = {0: 'airplane',
                    1: 'automobile',
                    2: 'bird',
                    3: 'cat',
                    4: 'deer',
                    5: 'dog',
                    6: 'frog',
                    7: 'horse',
                    8: 'ship',
                    9: 'truck'}

visualize = False

class DVSCifar10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, channels=2, multi_channel=1, T_prime=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.channels = channels
        self.multi_channel = multi_channel
        self.T_prime = T_prime
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()

    def __getitem__(self, index):
        data, target = torch.load(self.root + '/{}.pt'.format(index))

        if self.multi_channel:
            ## Bring down to T'
            data_tmp_list = []
            for i in range(self.T_prime):
                data_tmp = torch.zeros(2, 128, 128)
                chunks = int(data.shape[0] / self.T_prime)
                for chunk in range(chunks):
                    data_tmp = torch.logical_or(data_tmp, data[chunks * i + chunk, ...]).float()
                data_tmp_list.append(data_tmp)
            if self.T_prime != 0:
                data = torch.stack(data_tmp_list, dim=0)

            factor = 2 ** (self.channels - 1)
            compressed = torch.zeros(math.ceil(data.shape[0] / factor), 2, 128, 128)
            mc_data = torch.zeros(math.ceil(data.shape[0] / factor), self.channels, 2, 128, 128)

            padded_data = torch.nn.functional.pad(data, (0, 0, 0, 0, 0, 0, 0, data.shape[0] % factor), mode='constant',
                                                  value=0)

            for f in range(factor):
                compressed += padded_data[f::factor]

            compressed[compressed > 0] = 2 ** (torch.round(torch.log2(compressed[compressed > 0])))

            for c in range(self.channels):
                value = 2 ** (self.channels - c - 1)
                mc_data[:, self.channels - c - 1, ...] = torch.round(compressed / value)
                compressed[torch.round(compressed / value) == 1.0] = 0.0

            new_data = []
            for t in range(compressed.size(0)):
                mid_data = []
                for c in range(mc_data.size(1)):
                    mid_data.append(self.tensorx(self.resize(self.imgx(mc_data[t, c]))))
                new_data.append(torch.stack(mid_data, dim=0))

            if visualize:
                im_list = []
                for im in range(data.shape[0]):
                    im_list.append(self.imgx(data[im, 1, ...]))
                im_list[0].save("{}_{}_original.gif".format(mapping_dvscifar[target.item()], index), save_all=True,
                                append_images=im_list[1:], loop=0)

                for c in range(mc_data.size(1)):
                    im_list = []
                    for im in range(mc_data.size(0)):
                        im_list.append(self.imgx(mc_data[im, c, 1, ...]))
                    im_list[0].save("{}_{}_channel_{}.gif".format(mapping_dvscifar[target.item()], index, c),
                                    save_all=True,
                                    append_images=im_list[1:], loop=0)

                im_list = []
                for im in range(10):
                    img = torch.zeros(128, 128)
                    chunks = int(data.shape[0] / 10)
                    for chunk in range(chunks):
                        img = torch.logical_or(img, data[chunks * im + chunk, 1, ...]).float()
                    im_list.append(self.imgx(img))
                im_list[0].save("{}_{}_baseline.gif".format(mapping_dvscifar[target.item()], index), save_all=True,
                                append_images=im_list[1:], loop=0)

            data = torch.stack(new_data, dim=0)
            if self.transform is not None:
                flip = random.random() > 0.5
                if flip:
                    data = torch.flip(data, dims=(4,))
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(3, 4))

            if self.target_transform is not None:
                target = self.target_transform(target)
            return data, target.long().squeeze(-1)

        else:
            ## Bring down to T'
            data_tmp_list = []
            for i in range(self.T_prime):
                data_tmp = torch.zeros(2, 128, 128)
                chunks = int(data.shape[0] / self.T_prime)
                for chunk in range(chunks):
                    data_tmp = torch.logical_or(data_tmp, data[chunks * i + chunk, ...]).float()
                data_tmp_list.append(data_tmp)
            if self.T_prime != 0:
                data = torch.stack(data_tmp_list, dim=0)

            new_data = []
            for t in range(data.size(0)):
                new_data.append(self.tensorx(self.resize(self.imgx(data[t]))))

            data = torch.stack(new_data, dim=0)
            if self.transform is not None:
                flip = random.random() > 0.5
                if flip:
                    data = torch.flip(data, dims=(3,))
                off1 = random.randint(-5, 5)
                off2 = random.randint(-5, 5)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

            if self.target_transform is not None:
                target = self.target_transform(target)
            return data, target.long().squeeze(-1)

    def __len__(self):
        return len(os.listdir(self.root))


def build_dvscifar(path, channels, multi_channel, T_prime):
    train_path = path + '/train'
    val_path = path + '/test'
    train_dataset = DVSCifar10(root=train_path, transform=True, channels=channels, multi_channel=multi_channel,
                               T_prime=T_prime)
    val_dataset = DVSCifar10(root=val_path, channels=channels, multi_channel=multi_channel, T_prime=T_prime)

    return train_dataset, val_dataset


if __name__ == '__main__':
    batch_size = 16
    workers = 16

    train_dataset, val_dataset = build_dvscifar('./data/dvs-cifar10/', channels=3, multi_channel=1, T_prime=10)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=workers, pin_memory=True)
