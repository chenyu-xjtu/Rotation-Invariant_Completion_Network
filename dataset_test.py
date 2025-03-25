#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import torch
import open3d as o3d
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud

def load_data(partition):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    # f = h5py.File(os.path.join(DATA_DIR, 'MVP_Train_CP.h5'))
    # DATA_DIR = '/kaggle/input/multiview-partial-point-cloud-database'
    if(partition == 'train'):
        f = h5py.File(os.path.join('/home/chenyu/VGNet/data/MVP_Train_CP.h5'))
        # f = h5py.File(os.path.join('/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/chenyu/MVP/MVP_Train_CP.h5'))
    else:
        f = h5py.File(os.path.join('/home/chenyu/VGNet/data/MVP_Test_CP.h5'))
        # f = h5py.File(os.path.join('/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/chenyu/MVP/MVP_Test_CP.h5'))
    # print(f.keys())
    incomplete_data = f['incomplete_pcds'][::26]
    # (62400,2048,3)
    complete_data = f['complete_pcds'][:]
    # (2400,2048,3)
    label = f['labels'][0]
    return incomplete_data, complete_data, label

class MVP(Dataset):
    def __init__(self, prefix="train", npoints=2048, gaussian_noise=False,
                       unseen=False, factor=4):
        if prefix=="train":
            self.file_path = "/home/chenyu/VGNet/data/MVP_Train_CP.h5"
            # self.file_path = "/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/chenyu/MVP/MVP_Train_CP.h5"
            # self.file_path = '/home/lixiang/pcn_idam_verse/MVP/data/MVP_Train_CP.h5'
        elif prefix=="test":
            self.file_path = "/home/chenyu/VGNet/data/MVP_Test_CP.h5"
            # self.file_path = "/media/root/6701ae9d-6612-4271-8d50-522d7f72528b/chenyu/MVP/MVP_Test_CP.h5"
            # self.file_path = '/home/lixiang/pcn_idam_verse/MVP/data/MVP_Test_CP.h5'
        # elif prefix=="test":
        #     self.file_path = './data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")
        self.num_points = npoints #1024
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.factor = factor
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

    def __getitem__(self, item):
        label = (self.labels[item])
        # item为点云号
        incomplete_pointcloud = self.input_data[item]
        complete_pointcloud = self.gt_data[item // 26]
        #这里这样是将62400个残缺点云和2400个完整点云一起去训练了。如果是按load_data里的是将2400个残缺点云和2400个完整点云去训练
        # (2048,3)
        if self.gaussian_noise:
            incomplete_pointcloud = jitter_pointcloud(incomplete_pointcloud)
            complete_pointcloud = jitter_pointcloud(complete_pointcloud)
        if self.prefix != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        incomplete_pointcloud1 = incomplete_pointcloud.T  # incomplete_pointcloud1是未配准前的缺失点云
        complete_pointcloud1 = complete_pointcloud.T  # complete_pointcloud1是未配准前的完整点云

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        incomplete_pointcloud2 = rotation_ab.apply(incomplete_pointcloud1.T).T + np.expand_dims(translation_ab,
                                                                                                axis=1)  # incomplete_pointcloud2是配准后的缺失点云
        complete_pointcloud2 = rotation_ab.apply(complete_pointcloud1.T).T + np.expand_dims(translation_ab,
                                                                                            axis=1)  # complete_pointcloud2是配准后的完整点云

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        incomplete_pointcloud1 = np.random.permutation(incomplete_pointcloud1.T).T
        incomplete_pointcloud2 = np.random.permutation(incomplete_pointcloud2.T).T
        complete_pointcloud1 = np.random.permutation(complete_pointcloud1.T).T
        complete_pointcloud2 = np.random.permutation(complete_pointcloud2.T).T
        # 打乱各个点

        cal_normal = True
        if (cal_normal):
            incomplete_pointcloud1 = incomplete_pointcloud1.T
            incomplete_pointcloud2 = incomplete_pointcloud2.T
            complete_pointcloud1 = complete_pointcloud1.T
            complete_pointcloud2 = complete_pointcloud2.T

            incomplete_pointcloud1 = np.array(incomplete_pointcloud1) #(2048,3)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(incomplete_pointcloud1)
            # 估计点云法向量
            point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # 获取点云坐标和法向量数据
            points = np.asarray(point_cloud.points)
            normals = np.asarray(point_cloud.normals)
            incomplete_pointcloud1 = torch.cat((torch.from_numpy(points), torch.from_numpy(normals)), dim=1)

            incomplete_pointcloud2 = np.array(incomplete_pointcloud2) #(2048,3)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(incomplete_pointcloud2)
            # 估计点云法向量
            point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # 获取点云坐标和法向量数据
            points = np.asarray(point_cloud.points)
            normals = np.asarray(point_cloud.normals)
            incomplete_pointcloud2 = torch.cat((torch.from_numpy(points), torch.from_numpy(normals)), dim=1)

            # complete_pointcloud1 = np.array(complete_pointcloud1) #(2048,3)
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(complete_pointcloud1)
            # # 估计点云法向量
            # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # # 获取点云坐标和法向量数据
            # points = np.asarray(point_cloud.points)
            # normals = np.asarray(point_cloud.normals)
            # complete_pointcloud1 = torch.cat((torch.from_numpy(points), torch.from_numpy(normals)), dim=1)
            #
            # complete_pointcloud2 = np.array(complete_pointcloud2) #(2048,3)
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(complete_pointcloud2)
            # # 估计点云法向量
            # point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # # 获取点云坐标和法向量数据
            # points = np.asarray(point_cloud.points)
            # normals = np.asarray(point_cloud.normals)
            # complete_pointcloud2 = torch.cat((torch.from_numpy(points), torch.from_numpy(normals)), dim=1)

        #incomplete_pointcloud (B,2048,6)
        #complete_pointcloud (B,2048,3)
        return label, incomplete_pointcloud1, incomplete_pointcloud2, \
               complete_pointcloud1, complete_pointcloud2, \
               R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba

if __name__ == '__main__':
    # print(o3d)
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    # f = h5py.File(os.path.join(DATA_DIR, 'MVP_Train_CP.h5'))
    # incomplete_data = f['incomplete_pcds'][0]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(incomplete_data)
    # o3d.visualization.draw_geometries([pcd])
    train_loader = DataLoader(
        MVP(num_points=2048),
        batch_size=8, shuffle=True, partition='train', drop_last=True)
    test_loader = DataLoader(
        MVP(num_points=2048),
        batch_size=8, shuffle=True, partition='test', drop_last=True)
    for incomplete_pointcloud1, incomplete_pointcloud2, complete_pointcloud1, complete_pointcloud2,\
               R_ab, translation_ab, R_ba, translation_ba, euler_ab, euler_ba in tqdm(train_loader):
        incomplete_pointcloud1 = incomplete_pointcloud1.cuda() #(B, 3, 2048)
    print()
