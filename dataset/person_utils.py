import os
import cv2
import random
import numpy as np

from shutil import copy
from progress.bar import Bar

from .path import Path


# COCO
# mean = [0.46388339, 0.44664543, 0.41852783]
# std = [0.28203478, 0.27698355, 0.29013959]

# Finetune
mean = [0.468557, 0.45213289, 0.42389801]
std = [0.2804792, 0.27625015, 0.28853614]
NUM_CLASSES = 2


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def mean_std(path):
    '''
    Calculating the mean and standard deviation of a given dataset.
    '''
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


def _std(path, img_list, mean, pixels_num):
    '''
    Calculating the standard deviation of a given dataset.
    '''
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


def get_some_samples(data_path, save_path, num_images):
    img_path = os.path.join(data_path, 'Images')
    label_path = os.path.join(data_path, 'Masks')
    image_list = os.listdir(img_path)

    bar = Bar('Get some samples:', max=num_images)

    for i in range(num_images):
        name = image_list[i]
        copy(os.path.join(img_path, name), os.path.join(save_path, 'Images', name))
        copy(os.path.join(label_path, name), os.path.join(save_path, 'Masks', name))
        bar.suffix = f'{i + 1} / {num_images}'
        bar.next()
    bar.finish()


def load_subpaths(dataset_path):
    img_path = os.path.join(dataset_path, 'Images')
    mask_path = os.path.join(dataset_path, 'Masks')

    make_sure_path_exists(img_path)
    make_sure_path_exists(mask_path)
    return img_path, mask_path


def split_dataset(dataset_path, train_path, valid_path, valid_ratio):
    source_img_path, source_mask_path = load_subpaths(dataset_path)
    train_img_path, train_mask_path = load_subpaths(train_path)
    valid_img_path, valid_mask_path = load_subpaths(valid_path)

    file_list = os.listdir(source_img_path)
    total = len(file_list)
    offset = int(total * valid_ratio)

    if total == 0 or offset < 1:
        return [], file_list
    random.shuffle(file_list)

    valid_list, train_list = file_list[:offset], file_list[offset:]
    bar = Bar('Generating validset: ', max=offset)
    for i, img_file in enumerate(valid_list):
        copy(os.path.join(source_img_path, img_file), os.path.join(valid_img_path, img_file))
        copy(os.path.join(source_mask_path, img_file), os.path.join(valid_mask_path, img_file))
        bar.suffix = f'{i + 1} / {offset}'
        bar.next()
    bar.finish()

    bar = Bar('Generating trainset: ', max=total - offset)
    for i, img_file in enumerate(train_list):
        copy(os.path.join(source_img_path, img_file), os.path.join(train_img_path, img_file))
        copy(os.path.join(source_mask_path, img_file), os.path.join(train_mask_path, img_file))
        bar.suffix = f'{i + 1} / {total - offset}'
        bar.next()
    bar.finish()


if __name__ == '__main__':
    # data_path = Path.db_root_dir('medi')
    # train_path = os.path.join(Path.db_root_dir('finetune'), 'train')
    # valid_path = os.path.join(Path.db_root_dir('finetune'), 'val')

    # split_dataset(data_path, train_path, valid_path, 0.2)


    mode, num_images = 'train', 1000
    data_path = os.path.join(Path.db_root_dir('person'), mode)
    save_path = os.path.join(Path.db_root_dir('finetune'), mode)
    get_some_samples(data_path, save_path, num_images)

    # data_path = os.path.join(Path.db_root_dir('finetune'), 'train', 'Images')
    # mean, std = mean_std(data_path)
    # print('mean = ', mean, '\t', 'std = ', std)