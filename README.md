# Rotation-Invariant Completion Network
Our paper has been accepted by *PRCV* 2023 🚀🚀🚀

Please cite our paper [(pdf)](https://link.springer.com/chapter/10.1007/978-981-99-8432-9_10) if you find this code useful:
```
@InProceedings{10.1007/978-981-99-8432-9_10,
author="Chen, Yu
and Shi, Pengcheng",
editor="Liu, Qingshan
and Wang, Hanzi
and Ma, Zhanyu
and Zheng, Weishi
and Zha, Hongbin
and Chen, Xilin
and Wang, Liang
and Ji, Rongrong",
title="Rotation-Invariant Completion Network",
booktitle="Pattern Recognition and Computer Vision",
year="2024",
publisher="Springer Nature Singapore",
address="Singapore",
pages="115--127",
abstract="Real-world point clouds usually suffer from incompleteness and display different poses. While current point cloud completion methods excel in reproducing complete point clouds with consistent poses as seen in the training set, their performance tends to be unsatisfactory when handling point clouds with diverse poses. We propose a network named Rotation-Invariant Completion Network (RICNet), which consists of two parts: a Dual Pipeline Completion Network (DPCNet) and an enhancing module. Firstly, DPCNet generates a coarse complete point cloud. The feature extraction module of DPCNet can extract consistent features, no matter if the input point cloud has undergone rotation or translation. Subsequently, the enhancing module refines the fine-grained details of the final generated point cloud. RICNet achieves better rotation invariance in feature extraction and incorporates structural relationships in man-made objects. To assess the performance of RICNet and existing methods on point clouds with various poses, we applied random transformations to the point clouds in the MVP dataset and conducted experiments on them. Our experiments demonstrate that RICNet exhibits superior completion performance compared to existing methods.",
isbn="978-981-99-8432-9"
}
```
