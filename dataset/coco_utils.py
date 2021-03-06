import os
import numpy as np
import cv2

from shutil import copy
from progress.bar import Bar
from pycocotools.coco import COCO
from joblib import Parallel, delayed

from path import Path


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
 
 
def mask_generator(coco, width, height, anns_list):
    '''
    This function's purpose is generating the semantic segmentation mask of coco images.

    Args: 
        coco: COCO dataset tool object.
        width: Width of a specified image.
        height: Height of a specified image.
        anns_list: Annotations generated by pycocotools of a specified image.
    '''
    mask = np.zeros((height, width))

    for single in anns_list:
        mask_single = coco.annToMask(single)
        mask += mask_single

    mask[mask > 0] = 255
    return mask
 
 
def get_mask_data(ori_path, save_path, annFile, dataset, classes_names):
    '''
    This function's purpose is generating the semantic segmentation mask of coco images and
    saving them into specified paths individually.

    Args: 
        ori_path: Path to the COCO dataset.
        save_path: Path to the person segmentation dataset.
        annFile: A json file containing the annotations of COCO (train/valid) dataset.
        dataset: A flag indicating that the dataset is a training set or a validation set. 
                 It should be one of "train2017" or "val2017".
        classes_name: A class name which you wanna take out. It should be "person" in this task.
    '''
    ann_path = os.path.join(ori_path, 'annotations', annFile)

    coco = COCO(ann_path)
    classes_ids = coco.getCatIds(catNms=classes_names)
    imgIds_list = []

    for idx in classes_ids:
        imgidx = coco.getImgIds(catIds=idx)
        imgIds_list += imgidx

    imgIds_list = list(set(imgIds_list)) # Removing duplicate images.
    image_info_list = coco.loadImgs(imgIds_list)

    save_img_path = make_sure_path_exists(os.path.join(save_path, 'Images'))
    save_mask_path = make_sure_path_exists(os.path.join(save_path, 'Masks'))

    num_imgs = len(image_info_list)
    bar = Bar('generating the mask of coco images: ', max=num_imgs)
 
    for i, imageinfo in enumerate(image_info_list):
        annIds = coco.getAnnIds(imgIds=imageinfo['id'], catIds=classes_ids, iscrowd=None) # Get the segmentation information of the corresponding category.
        anns_list = coco.loadAnns(annIds)

        file_name = imageinfo['file_name']
        mask = mask_generator(coco, imageinfo['width'], imageinfo['height'], anns_list)

        copy(os.path.join(ori_path, dataset, file_name), os.path.join(save_img_path, file_name))
        cv2.imwrite(os.path.join(save_mask_path, file_name), mask)

        bar.suffix = f'{i+1} / {num_imgs}'
        bar.next()
    bar.finish()


def imgs2arrays(source_path, target_path):
    img_list = os.listdir(source_path)
    num_imgs = len(img_list)
    bar = Bar('Transposing images to numpyarrays: ', max=num_imgs)
    Parallel(n_jobs=8)(delayed(img2array)(source_path, target_path, img_file, i, num_imgs, bar) \
                                            for i, img_file in enumerate(img_list))
    bar.finish()
    print('Trasnposing mission complete.')


def img2array(source_path, target_path, img_file, i, num_imgs, bar):
    img = cv2.imread(os.path.join(source_path, img_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    np.save(os.path.join(target_path, img_file[:-4]), img)
    bar.suffix = f'{i + 1} / {num_imgs}'
    bar.next()


def trans_samples_to_array(source_path, target_path, mode='train'):
    img_path = make_sure_path_exists(os.path.join(target_path, mode, 'Images_np'))
    ann_path = make_sure_path_exists(os.path.join(target_path, mode, 'Masks_np'))

    imgs2arrays(os.path.join(source_path, mode, 'Images'), img_path)
    imgs2arrays(os.path.join(source_path, mode, 'Masks'), ann_path)

 
if __name__ == '__main__':
    ori_path = Path.db_root_dir('coco')
    root_save_path = Path.db_root_dir('person')

    classes_names = ['person']
    datasets_list = ['train2017', 'val2017']

    for dataset in datasets_list:
        save_path = os.path.join(root_save_path, f'{dataset[:-4]}')
        annFile = f'instances_{dataset}.json'
 
        get_mask_data(ori_path, save_path, annFile, dataset, classes_names)
        print('Got all the masks of "{}" from {}'.format(classes_names, dataset))