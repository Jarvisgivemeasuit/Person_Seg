import os
from torch.utils.data import Dataset

import cv2
import albumentations as A

from .person_utils import *
from .path import Path


class PersonSeg(Dataset):
    '''
    Loading the specified dataset into Pytorch Dataset iterator.
    '''
    NUM_CLASSES = 2
    
    def __init__(self, mode='train', base_dir=Path.db_root_dir('person')):
        assert mode in ['train', 'val']
        super().__init__()

        self.mode = mode
        self.mean = mean
        self.std = std

        self.jpg_image_dir = os.path.join(base_dir, mode, 'Images')
        self.jpg_label_dir = os.path.join(base_dir, mode, 'Masks')

        self.array_image_dir = os.path.join(base_dir, mode, 'Images_np')
        self.array_label_dir = os.path.join(base_dir, mode, 'Masks_np')

        self._data_list = os.listdir(self.jpg_image_dir)

        for data in self._data_list:
            if data[-3:] != 'jpg':
                self._data_list.remove(data)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self.load_sample(idx)

    def load_img_sample(self, idx):
        image = cv2.imread(os.path.join(self.jpg_image_dir, self._data_list[idx]))
        mask = cv2.imread(os.path.join(self.jpg_label_dir, self._data_list[idx]), -1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask[mask > 0] = 1
        mask = mask.astype(np.float)

        sample = {'image': image, 'label': mask}
        if self.mode == 'train':
            sample = self._train_enhance(sample)
        else:
            sample = self._test_enhance(sample)

        sample['image'] = sample['image'].transpose((2, 0, 1))
        sample['file'] = self._data_list[idx]

        return sample

    def load_array_sample(self, idx):
        image = np.load(os.path.join(self.array_image_dir, self._data_list[idx]))
        mask = np.load(os.path.join(self.array_label_dir, self._data_list[idx]))

        mask[mask > 0] = 1
        mask = mask.astype(np.float)

        sample = {'image': image, 'label': mask}
        if self.mode == 'train':
            sample = self._train_enhance(sample)
        else:
            sample = self._test_enhance(sample)

        sample['image'] = sample['image'].transpose((2, 0, 1))
        sample['file'] = self._data_list[idx]

        return sample

    def _train_enhance(self, sample):
        compose = A.Compose([
            A.Resize(256, 256, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ElasticTransform(p=0.5),
            A.Blur(p=0.5),
            A.CoarseDropout(p=0.5),

            A.Normalize(mean=self.mean, std=self.std, p=1),
        ], additional_targets={'image': 'image', 'label': 'mask'})
        return compose(**sample)

    def _test_enhance(self, sample):
        norm = A.Compose([
            A.Resize(256, 256, p=1),
            A.Normalize(mean=self.mean, std=self.std, p=1)],
            additional_targets={'image': 'image', 'label': 'mask'})
        return norm(**sample)