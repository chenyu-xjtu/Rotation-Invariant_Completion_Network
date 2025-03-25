import h5py
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import reshape
import open3d as o3d

f = h5py.File('/home/lixiang/pcn_idam_verse/MVP/data/MVP_Test_RG.h5','r')
f.keys() #可以查看所有的主键
print([key for key in f.keys()])
print(f['cat_labels'].shape)
print(f['complete'].shape)
print(f['match_id'])
print(f['match_level'].shape)
# print(f['pose_src'][:].shape)
# print(f['pose_tgt'][:].shape)
# print(f['rot_level'][:].shape)
print(f['rotated_src'][:].shape)
print(f['rotated_tgt'][:].shape)
print(f['src'].shape)
print(f['tgt'].shape)
# print(f['transforms'][:].shape)
complete_data = np.array(f['complete'][()])
src = np.array(f['src'][()])
tgt = np.array(f['tgt'][()])
rotated_src = np.array(f['rotated_src'][()])
rotated_tgt = np.array(f['rotated_tgt'][()])

p1 = np.array(complete_data[30])  # (2048,3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(p1)
points = np.asarray(point_cloud.points)
colors = None
ax = plt.axes(projection='3d')
ax.view_init(90, -90)
ax.axis("off")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
plt.show()

p1 = np.array(src[30])  # (2048,3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(p1)
points = np.asarray(point_cloud.points)
colors = None
ax = plt.axes(projection='3d')
ax.view_init(90, -90)
ax.axis("off")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
plt.show()

p1 = np.array(tgt[30])  # (2048,3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(p1)
points = np.asarray(point_cloud.points)
colors = None
ax = plt.axes(projection='3d')
ax.view_init(90, -90)
ax.axis("off")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
plt.show()

p1 = np.array(rotated_src[30])  # (2048,3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(p1)
points = np.asarray(point_cloud.points)
colors = None
ax = plt.axes(projection='3d')
ax.view_init(90, -90)
ax.axis("off")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
plt.show()

p1 = np.array(rotated_tgt[30])  # (2048,3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(p1)
points = np.asarray(point_cloud.points)
colors = None
ax = plt.axes(projection='3d')
ax.view_init(90, -90)
ax.axis("off")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=colors)
plt.show()





