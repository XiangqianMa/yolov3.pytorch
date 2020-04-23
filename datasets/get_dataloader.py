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
        image_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        dataset_format='coco',
        train_augmentation=None,
        val_augmentation=None,
        dataset="coco",
        normalize=False,
        multi_scale=False
    ):
        self.dataset = dataset

        self.train_images_root = train_images_root
        self.train_annotations_root = train_annotations_root
        self.train_augmentation = train_augmentation
        
        self.val_images_root = val_images_root
        self.val_annotations_root = val_annotations_root
        self.val_augmentation = val_augmentation

        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.normalize = normalize
        self.multi_scale = multi_scale
        self.dataset_format = dataset_format
    
    def get_dataloader(self, batch_size):
        train_augment = self.__get_data_augment__(self.train_augmentation)
        train_datset = self.__get_dataset__(self.train_images_root, self.train_annotations_root, train_augment,
                                            True, self.multi_scale)
        val_augment = self.__get_data_augment__(self.val_augmentation)
        val_dataset = self.__get_dataset__(self.val_images_root, self.val_annotations_root, val_augment,
                                           False, False)

        train_dataloader = DataLoader(train_datset, batch_size=batch_size, num_workers=8, pin_memory=True,
                                      shuffle=True, collate_fn=train_datset.collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True,
                                    shuffle=False, collate_fn=val_dataset.collate_fn)

        return train_dataloader, val_dataloader

    def __get_data_augment__(self, augmentation):
        augment = None
        if augmentation is not None:
            augment = DataAugment(aug=augmentation, dataset_format=self.dataset_format)
        return augment
    
    def __get_dataset__(self, images_root, annotations_root, augment, augment_flag=True, multi_scale=True):
        dataset = None
        print("@ Dataset: %s." % self.dataset)
        if self.dataset == 'official':
            dataset = ListDataset(images_root, annotations_root, self.image_size, augment_flag, multi_scale)
        elif self.dataset == 'coco':
            dataset = COCODataset(images_root, annotations_root, self.image_size, self.mean,
                                  self.std, augment=augment,
                                  normalize=self.normalize, multi_scale=multi_scale)
        elif self.dataset == 'oxfordhand':
            dataset = OxfordDataset(images_root, annotations_root, self.image_size, augment_flag, multi_scale)
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
