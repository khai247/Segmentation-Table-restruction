import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from utilss import create_mask
import glob
from torch.utils.data import DataLoader

class SegmentDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        #mask_path = os.path.join(self.mask_dir, self.images[index])
        mask_path = img_path.replace("image", "mask").replace("png", "txt")

        image = np.array(Image.open(img_path))
        img_h, img_w, _ = image.shape

        mask = create_mask(img_h=img_h,img_w=img_w,label_path=mask_path,LINE_HEIGHT_RATIO=0.5)


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        #assert image.shape[:2] == mask.shape[:2]
        return image, mask
def get_loader(
            train_dir,
            train_maskdir,
            val_dir,
            val_maskdir,
            batch_size,
            train_transform,
            val_transform,
            num_workers=4,
            pin_memory=True):
    train_ds = SegmentDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform= train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = SegmentDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return train_loader,val_loader