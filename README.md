<div id="top" align="center">
  
#  <p align="center">基于混合注意力机制的轻量化自监督单目深度估计</p>

<p align="center">刘佳, 王艺阳, 丁帅, 卢国瑞, 陈大鹏*</p>
<p align="center">南京信息工程大学</p>
</div>

## <p align="center">摘要</p>
针对目前单目深度估计模型计算复杂度高、参数量大等问题, 结合改进的卷积神经网络与混合注意力机制, 构建了轻量级的自监督单目深度估计模型. 模型将线性注意力机制加入到网络编码器, 并结合扩张卷积和可变形卷积网络模块, 增强对多尺度特征的提取能力, 同时有效降低计算和内存消耗. 在解码器与编码器的跳跃式连接部分, 加入双通道注意力机制, 有效提升低层和高层特征的融合效果, 增强了深度估计的鲁棒性和精度. 实验结果表明, 所提方法在KITTI数据集上的AbsRel为0.101, 模型参数量仅为3.0M, 推理速度达到2.8ms. 所提方法在保证高精度的同时, 显著降低了计算复杂度, 具有较强的实时性和实用性, 适合应用于边缘设备和实时深度估计任务. 


## 概述
<img src="./img/Figure_1.jpg" width="100%" alt="overview" align=center />

## KITTI数据集可视化的对比
<img src="./img/Figure_2.jpg" width="100%" alt="overview" align=center />

## KITTI数据集可视化的比较 
| Model                                | Parameters (M) | AbsRel | SqRel | RMSE  | RMSElog | δ1   | δ2   | δ3   |
|--------------------------------------|----------------|--------|-------|-------|---------|-------|-------|-------|
| Zhou等                               | 34.2           | 0.208  | 1.768 | 6.958 | 0.283   | 0.678 | 0.885 | 0.957 |
| SGDepth                              | 16.3           | 0.113  | 0.835 | 4.693 | 0.191   | 0.879 | 0.961 | 0.981 |
| MonoFormer-ViT                       | 23.9           | 0.108  | 0.806 | 4.594 | 0.184   | 0.884 | 0.963 | 0.983 |
| Monodepth2                           | 32.5           | 0.115  | 0.903 | 4.863 | 0.193   | 0.877 | 0.959 | 0.981 |
| R-MSMF6                              | 3.8            | 0.126  | 0.944 | 4.981 | 0.204   | 0.857 | 0.952 | 0.978 |
| Lite-Mono                            | 3.1            | 0.107  | 0.765 | **4.461** | 0.183   | 0.886 | 0.963 | 0.983 |
| HR-Depth                             | 14.7           | 0.109  | 0.792 | 4.632 | 0.185   | 0.884 | 0.962 | 0.981 |
| Sc-depthv3                           | 59.3           | 0.118  | 0.756 | 4.756 | 0.188   | 0.864 | 0.960 | 0.980 |
| DNA-Depth-B0                         | 4.2            | 0.105  | 0.748 | 4.489 | 0.179   | 0.892 | 0.965 | 0.981 |
| Wang等                               | 28.3           | 0.106  | 0.802 | 4.538 | 0.186   | 0.853 | 0.949 | 0.977 |
| **Ours**                             | **3.0**        | **0.101** | **0.746** | 4.543 | **0.178** | **0.896** | **0.966** | **0.985** |



## Data Preparation
请参考 [Monodepth2](https://github.com/nianticlabs/monodepth2) 准备您的KITTI数据.

## Install

模型是用 CUDA 11.8, Python 3.9.x (conda environment), 和 PyTorch 2.4.1 训练的.

使用PyTorch库创建conda环境:

```bash
conda create -n LSMDepth python=3.9.4
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate LSMDepth
```

安装requirements.txt中列出的必备软件包:
```bash
pip install -r requirements.txt
```

## 训练
通过运行KITTI数据集，可以对模型进行训练:
```bash
python train.py --data_path path/to/your/data --model_name mymodel
```

## 推理
要对单个图像进行推理，请运行:
```bash
python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image
```
## 评估
要在KITTI上评估模型，请运行:
```bash
python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --data_path path/to/kitti_data/ --model lite-mono
```
