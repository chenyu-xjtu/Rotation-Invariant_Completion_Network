import torch
import numpy as np
import torch.utils.data as data
import h5py
import os

class MVP(data.Dataset):
    def __init__(self, prefix="train", npoints=2048):
        if prefix=="train":
            # self.file_path = "/home/chenyu/VGNet/data/MVP_Train_CP.h5"
            # self.file_path = "/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/chenyu/MVP/MVP_Train_CP.h5"
            self.file_path = '/home/lixiang/pcn_idam_verse/MVP/data/MVP_Train_CP.h5'
        elif prefix=="test":
            # self.file_path = "/home/chenyu/VGNet/data/MVP_Test_CP.h5"
            # self.file_path = "/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/chenyu/MVP/MVP_Test_CP.h5"
            self.file_path = '/home/lixiang/pcn_idam_verse/MVP/data/MVP_Test_CP.h5'
        # elif prefix=="test":
        #     self.file_path = './data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")
        self.prefix = prefix
        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])
        print(self.input_data.shape)
        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        print(self.gt_data.shape, self.labels.shape)
        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])
        return label, partial, complete