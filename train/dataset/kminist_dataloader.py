"""
Create train, valid, main iterators for CIFAR-10 [1].
Easily extended to KMNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import os
import numpy as np
# from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as tdata

from train.dataset.dataloader import BaseDataloader
from train.dataset.cutout import Cutout


class KMNIST(BaseDataloader):
    def __init__(self):
        super(KMNIST, self).__init__()
        self.root = os.path.expanduser('~/dataset/kmnist/')
        self.input_size = [28, 28, 1]
        self.out_cls_num = 10

    def get_train_dataloader(self):
        if self.valid_size is not None:
            self.train_dataloader, self.val_dataloader = \
                self.__get_train_valid_loader(self.root,
                                              self.batch_size,
                                              self.augment,
                                              self.random_seed,
                                              self.valid_size,
                                              self.shuffle,
                                              self.show_sample,
                                              self.num_workers,
                                              self.pin_memory)
        else:
            self.train_dataloader = self.__get_train_loader(self.root,
                                                            self.batch_size,
                                                            self.shuffle,
                                                            self.num_workers,
                                                            self.pin_memory)

        return self.train_dataloader

    def get_val_dataloader(self):
        if self.val_dataloader is None:
            self.val_dataloader = self.__get_test_loader(self.root,
                                                         self.batch_size,
                                                         self.shuffle,
                                                         self.num_workers,
                                                         self.pin_memory)
        return self.val_dataloader

    def get_test_dataloader(self):
        return self.__get_test_loader(self.root,
                                      self.batch_size,
                                      self.shuffle,
                                      self.num_workers,
                                      self.pin_memory)

    def __get_train_valid_loader(self,
                                 data_dir,
                                 batch_size,
                                 augment,
                                 random_seed,
                                 valid_size=0.2,
                                 shuffle=True,
                                 show_sample=False,
                                 num_workers=4,
                                 pin_memory=False):
        """
        Utility function for loading and returning train and valid
        multi-process iterators over the KMNIST dataset. A sample
        9x9 grid of the images can be optionally displayed.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - augment: whether to apply the data augmentation scheme
        mentioned in the paper. Only applied on the train split.
        - random_seed: fix seed for reproducibility.
        - valid_size: percentage split of the training set used for
        the validation set. Should be a float in the range [0, 1].
        - shuffle: whether to shuffle the train/validation indices.
        - show_sample: plot 9x9 sample grid of the dataset.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        Returns
        -------
        - train_loader: training set iterator.
        - valid_loader: validation set iterator.
        """
        error_msg = "[!] valid_size should be in the range [0, 1]."
        assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

        normalize = transforms.Normalize(
            mean=[0.5],
            std=[0.5],
        )

        # define transforms
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        if augment:
            '''
            train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
             #   transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            '''
            train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
          #      transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),  # R,G,B每层的归一化用到的均值和方差
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        # load the dataset
        train_dataset = datasets.KMNIST(
            root=data_dir, train=True,
            download=self.download, transform=train_transform,
        )

        valid_dataset = datasets.KMNIST(
            root=data_dir, train=True,
            download=self.download, transform=valid_transform,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            # np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = tdata.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = tdata.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return train_loader, valid_loader

    def __get_train_loader(self,
                           data_dir,
                           batch_size,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=False):
        """
        Utility function for loading and returning a multi-process
        main iterator over the KMNIST dataset.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        Params
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - shuffle: whether to shuffle the dataset after every epoch.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        Returns
        -------
        - data_loader: main set iterator.
        """
        normalize = transforms.Normalize(
            mean=[0.5],
            std=[0.5],
        )

        # define transform
        '''
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        '''

        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
         #   transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # R,G,B每层的归一化用到的均值和方差
        ])

        dataset = datasets.KMNIST(
            root=data_dir, train=True,
            download=self.download, transform=train_transform,
        )

        data_loader = tdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return data_loader

    def __get_test_loader(self,
                          data_dir,
                          batch_size,
                          shuffle=True,
                          num_workers=1,
                          pin_memory=False):
        """
        Utility function for loading and returning a multi-process
        main iterator over the KMNIST dataset.
        If using CUDA, num_workers should be set to 1 and pin_memory to True.
        ParamsZ
        ------
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - shuffle: whether to shuffle the dataset after every epoch.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
        True if using GPU.
        Returns
        -------
        - data_loader: main set iterator.
        """
        normalize = transforms.Normalize(
            mean=[0.5],
            std=[0.5],
        )

        # define transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        dataset = datasets.KMNIST(
            root=data_dir, train=False,
            download=self.download, transform=transform,
        )

        data_loader = tdata.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        return data_loader



if __name__ == '__main__':
    dataset = 'MNIST'
    root_path = os.path.expanduser('~/dataset/mnist/')
    ls_dataset = ['MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST']
    if dataset in ls_dataset:
        from comm.registry import Registry
        dataloader_cls = Registry.DataLoaderRegistry.query(dataset)
        dataloader_cls_ins = dataloader_cls()
    else:
        from train.dataset.comm_data import FDataLoader
        dataloader_cls_ins = FDataLoader()
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainData = datasets.CIFAR100(root=root_path,
                      train=True,
                      transform=transform,
                      download=True)
    testData = datasets.CIFAR100(root=root_path,
                     train=False,
                     transform=transform,
                     download=True)

    batch_size = 64
    trainData_loader = tdata.DataLoader(dataset=trainData,
                                  batch_size=batch_size,
                                  shuffle=True)

    testData_loader = tdata.DataLoader(dataset=testData,
                                 batch_size=batch_size,
                                 shuffle=True)

    examples = enumerate(trainData_loader)
    idx, (data, labels) = next(examples)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 4, i * 4 + j + 1)
            plt.imshow(data[i * 4 + j].permute(1,2,0))
            plt.xticks([])
            plt.yticks([])
    plt.savefig('figure4_4.png')
    plt.show()

#