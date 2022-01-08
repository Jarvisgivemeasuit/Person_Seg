import os

BASE_PATH = '/home/lijl/Datasets'

class Path:
    @staticmethod
    def db_root_dir(dataset_name):
        if dataset_name == 'coco':
            return os.path.join(BASE_PATH, 'detection', 'MS_COCO_dataset')
        elif dataset_name == 'person':
            return os.path.join(BASE_PATH, 'segmentation', 'coco_person')
        elif dataset_name == 'medi':
            return os.path.join(BASE_PATH, 'segmentation', 'medical_room')
        elif dataset_name == 'finetune':
            return os.path.join(BASE_PATH, 'segmentation', 'finetune')
        else:
            print('Dataset {} not available.'.format(dataset_name))
            raise NotImplementedError