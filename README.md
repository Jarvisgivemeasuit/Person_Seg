# Person_Seg

任务: 判断室内是否有人

解决方法: 语义分割

## 目录说明
### dataset

该目录包含训练前对数据集的所有相关操作。具体包括数据集的制作、数据集均值与标准差的统计、数据增强策略以及用pytorch数据加载等；
- path.py

    包含所有数据集存放路径。

- coco_utils.py

    数据集取自公共数据集COCO中所有包含人的实例分割的样本。该文件主要实现将指定类别图片保存至目标目录以及图片对应的mask制作。
    数据集制作完成后的文件目录如下：
    
```
    coco_person
        ├── test
        │   └── Images
        ├── train
        │   ├── Images
        │   └── Masks
        └── val 
            ├── Images
            └── Masks
```
- person_dataset.py

    将指定的数据集加载到Pytorch数据集迭代器并进行数据增强，其中数据增强使用工具为Albumentations。

- person_utils.py

    统计数据集的均值与标准差。

### model

该目录包含模型训练使用的模型。
- __init__.py

    模型定义与模型保存。

- PSPnet

    - mobilenet_v3.py

        摘自torchvision。

    - pspnet.py

        以mobilenet_v3为backbone的pspnet模型。`parameters size=4.89M`

### utils

该目录包含所有训练超参数、评价指标及后处理等问题。
- args.py

    设置训练过程中的所有超参数。

- metrics.py

    所有模型训练涉及的评价指标及平均数计算（如单位iter训练时间、loss等）。

- utils.py

    设计weight decay策略等模型参数分离、lr下降策略等。

### main.py

训练过程pipline以及验证集效果可视化。

### inference.py

对训练好的模型进行推理。
