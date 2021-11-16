# Person_Seg

判断室内是否有人。经考虑后使用语义分割解决该问题。

## 目录说明
### dataset

该目录包含训练前对数据集的所有相关操作。具体包括数据集的制作、数据集均值与标准差的统计、数据增强策略以及用pytorch数据加载等；
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
