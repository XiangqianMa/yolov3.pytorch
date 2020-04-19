from datasets.coco_dataset import COCODataset
from datasets.official_dataset import ListDataset
from datasets.oxfordhand import OxfordDataset
from datasets.data_augment import DataAugment
from torch.utils.data import DataLoader


class GetDataLoader(object):
    def __init__(
        self, 
        train_images_root, 
        train_annotations_root, 
        val_images_root,
        val_annotations_root,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        dataset_format='coco',
        train_augmentation={'Resize': {'height': 416, 'width':416, 'always_apply': True}},
        val_augmentation={'Resize': {'height': 416, 'width':416, 'always_apply': True}},
        dataset="coco"
    ):
        self.dataset = dataset

        self.train_images_root = train_images_root
        self.train_annotations_root = train_annotations_root
        self.train_augmentation = train_augmentation
        
        self.val_images_root = val_images_root
        self.val_annotations_root = val_annotations_root
        self.val_augmentation = val_augmentation

        self.mean = mean
        self.std = std
        self.dataset_format = dataset_format

        self.train_transform = self.__get_data_augment__(train_augmentation)
        self.val_transform = self.__get_data_augment__(val_augmentation)
    
    def get_dataloader(self, batch_size):
        train_datset = self.__get_datasets__(self.train_images_root, self.train_annotations_root, self.train_transform)
        val_dataset = self.__get_datasets__(self.val_images_root, self.val_annotations_root, self.val_transform)

        train_dataloader = DataLoader(train_datset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True, collate_fn=train_datset.collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False, collate_fn=val_dataset.collate_fn)

        return train_dataloader, val_dataloader

    def __get_data_augment__(self, augmentation):
        return DataAugment(aug=augmentation, dataset_format=self.dataset_format)
    
    def __get_datasets__(self, images_root, annotations_root, transform):
        dataset = None
        if self.dataset == 'official':
            dataset = ListDataset(images_root, annotations_root, 416, True, True)
        elif self.dataset == 'coco':
            dataset = COCODataset(images_root, annotations_root, self.mean, self.std, transforms=transform)
        elif self.dataset == 'oxfordhand':
            dataset = OxfordDataset(images_root, annotations_root, 416, True, False)
        return dataset


if __name__ == '__main__':
    train_root = 'data/coco/val2017'
    train_annotation_root = 'data/coco/val2017_txt'
    val_root = 'data/coco/val2017'
    val_annotation_root = 'data/coco/val2017_txt'
    get_dataloader = GetDataLoader(train_root, train_annotation_root, val_root, val_annotation_root)
    train_dataloader, val_dataloader = get_dataloader.get_dataloader(8)
    for paths, images, targets in train_dataloader:
        print('--')
    pass
