<div id="top" align="center">
  
# LSMDepth 
**Lightweight self-supervised monocular depth estimation based on hybrid attention mechanism**
  
  Jia Liu, Guorui Lu, Jiaxu Ning, Lina Wei, Dapeng Chen*
  
</div>


## Overview
<img src="./img/Figure_1.jpg" width="100%" alt="overview" align=center />

## KITTI Results
<img src="./img/Figure_2.jpg" width="100%" alt="overview" align=center />

## Comparison of KITTI dataset results
| Model                                | Parameters (M) | AbsRel | SqRel | RMSE  | RMSElog | δ1   | δ2   | δ3   |
|--------------------------------------|----------------|--------|-------|-------|---------|-------|-------|-------|
| Zhou                                 | 34.2           | 0.208  | 1.768 | 6.958 | 0.283   | 0.678 | 0.885 | 0.957 |
| SGDepth                              | 16.3           | 0.113  | 0.835 | 4.693 | 0.191   | 0.879 | 0.961 | 0.981 |
| MonoFormer-ViT                       | 23.9           | 0.108  | 0.960 | 4.594 | 0.184   | 0.884 | 0.950 | 0.981 |
| Monodepth2                           | 32.5           | 0.115  | 0.903 | 4.863 | 0.193   | 0.877 | 0.959 | 0.981 |
| R-MSMF6                              | 3.8            | 0.120  | 1.062 | 5.800 | 0.204   | 0.857 | 0.948 | 0.978 |
| Lite-Mono                            | 3.1            | 0.107  | 0.765 | 4.461 | 0.183   | 0.886 | 0.960 | 0.979 |
| MonoViT-tiny                         | 10.3           | 0.106  | 0.749 | 4.484 | 0.183   | 0.888 | 0.961 | 0.980 |
| HR-Depth                             | 14.7           | 0.109  | 0.792 | 4.632 | 0.185   | 0.884 | 0.959 | 0.979 |
| Sc-depth3                            | 59.3           | 0.118  | 0.756 | 4.756 | 0.188   | 0.844 | 0.960 | 0.980 |
| DNA-Depth-B0                         | 9.1            | 0.130  | 1.053 | 5.144 | 0.208   | 0.853 | 0.940 | 0.979 |
| Bian                                 | 7.0            | 0.125  | 0.856 | 5.071 | 0.201   | 0.849 | 0.948 | 0.980 |
| **Ours**                             | **3.0**        | **0.102** | **0.746** | **4.543** | **0.178** | **0.896** | **0.964** | **0.983** |



## Data Preparation
Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) to prepare your KITTI data.

## Single Image Test
    python test_simple.py --load_weights_folder path/to/your/weights/folder --image_path path/to/your/test/image

## Training
    python train.py --data_path path/to/your/data --model_name mymodel
