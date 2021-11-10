import os
import cv2
import numpy as np
from progress.bar import Bar

from .path import Path


mean = [0.46388339, 0.44664543, 0.41852783]
std = [0.28203478, 0.27698355, 0.29013959]
NUM_CLASSES = 2


# 统计类别数量占比
def statistic(data_path):
    data_list = os.listdir(os.path.join(data_path, 'mask'))
    num = len(data_list)
    bar = Bar('counting:', max=num)
    res = np.zeros(16)
    for idx, data_file in enumerate(data_list):
        mask = np.load(os.path.join(data_path, 'mask', data_file))
        for i in range(16):
            count = (mask == i).sum()
            res[i] += count
            
        bar.suffix = '{} / {}'.format(idx, num)
        bar.next()
    bar.finish()
    return res


# 将多类别label转换成前背景两类mask
def fore_back(path_dict):
    img_list = os.listdir(path_dict['data_path'])
    num_imgs = len(img_list)
    bar = Bar('Saving binary file:', max=num_imgs)

    for i, mask_file in enumerate(img_list):
        mask = np.load(os.path.join(path_dict['data_path'], mask_file))

        back = (mask == 15).sum()
        rate = (mask.size - back) / mask.size

        binary = np.ones(mask.shape)
        binary[np.where(mask == 15)] = 0

        np.save(os.path.join(path_dict['save_path'], mask_file), {'binary_mask': binary, 'rate': rate})
        
        bar.suffix = f'{i + 1} / {num_imgs}'
        bar.next()
    bar.finish()


#  计算所有图片像素的均值并调用std
def mean_std(path):
    img_list = os.listdir(path)
    pixels_num = 0
    value_sum = [0, 0, 0]
    files_num = len(img_list)
    bar = Bar('Calculating mean:', max=files_num)

    for i, img_file in enumerate(img_list):
        img = cv2.imread(os.path.join(path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        pixels_num += img.shape[0] * img.shape[1]
        value_sum += np.sum(img, axis=(0, 1))

        bar.suffix = f'{i}/{files_num}'
        bar.next()
    bar.finish()

    value_mean = value_sum / pixels_num
    print('mean = ', value_mean)
    value_std = _std(path, img_list, value_mean, pixels_num)
    return value_mean, value_std


# 计算所有图片的标准差
def _std(path, img_list, mean, pixels_num):
    files_num = len(img_list)
    bar = Bar('Calculating std:', max=files_num)
    value_std = [0, 0, 0]
    i = 0
    for img_file in img_list:
        img = cv2.imread(os.path.join(path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        value_std += np.sum((img - mean) ** 2, axis=(0, 1))
        i += 1
        bar.suffix = f'{i}/{files_num}'
        bar.next()
    bar.finish()
    return np.sqrt(value_std / pixels_num)



if __name__ == '__main__':
    data_path = os.path.join(Path.db_root_dir('person'),'train', 'Images')
    mean, std = mean_std(data_path)
    print('mean = ', mean, '\t', 'std = ', std)

    # dis_path = os.path.join(paths_dict['train_split_256'], 'mask')
    # distributing(dis_path)