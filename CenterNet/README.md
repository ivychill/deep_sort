## 数据准备

### 实验路径组织：

|- root_dir \
&emsp;|- data \
&emsp;&emsp; |- coco \
&emsp;&emsp;&emsp; |- annotations \
&emsp;&emsp;&emsp; |- images \
&emsp;&emsp; |- drone \
&emsp;&emsp;&emsp; |- annotations \
&emsp;&emsp;&emsp; |- images \
&emsp;|- exp \
&emsp;&emsp;|- exp_id_sub_dir

data路径下放置coco和drone两个数据集的文件。annotations文件夹内放置json文件，其中存储图片标注信息，包含的键值对参考coco官方格式；
images文件夹中存放标注信息中引用到的图片。

exp路径下存放实验的结果文件，运行训练后会自动创建实验子文件夹。

## Detector训练过程

### 1. 首先在coco数据集上预训练

修改`./src/lib/datasets/dataset/coco.py`中的`self.img_dir`路径为实验路径组织中要求的`coco/images`路径，`load_model`为加载的
预训练模型`ExtremeNet_500000.pth`的路径，`train_annot_path`为训练集标注文件，`test_annot_path`为测试集标注文件，每跑完一个epoch
便在测试数据上进行一轮测试，便于筛选最优模型。

执行命令：`./pretrain_hg_coco.sh`。

### 2. 然后在drone数据集上再训练

修改`./src/lib/datasets/dataset/drone.py`中的`self.img_dir`路径为实验路径组织中要求的`drone/images`路径，`load_model`为加载的
在coco数据集上预训练得到的模型的路径，`train_annot_path`为训练集标注文件，`test_annot_path`为测试集标注文件，每跑完一个epoch
便在测试数据上进行一轮测试，便于筛选最优模型。

执行命令：`./train_hg_drone.sh`。