import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2

import glob
from torch.utils.data import DataLoader

def create_mask(img_h, img_w, label_path,LINE_HEIGHT_RATIO):
    line_mask = np.zeros((img_h, img_w))

    with open(label_path, 'r') as f:

        label_lines = f.readlines()
        # draw line mask
        for line in label_lines:
            line_content = line.strip().split(' ') if '[' not in line else ['[' + a for a in line.strip().split('[')]
            line_type = line_content[0]
            if 'line' in line_type in line_type:
                if len(line_content) == 5: # type x,y,w,h
                    x, y, w, h = [int(num) for num in line_content[1:]]
                    line_mask[y:y+h, x:x+w] = 255
                elif len(line_content) == 3: # type x_points, y_points
                    if ',' in line_content[1].strip()[1:-1]:
                        sep = ', '
                    else:
                        sep = ' '
                    x_points, y_points = np.fromstring(line_content[1].strip()[1:-1], sep=sep).astype(
                        np.int32), np.fromstring(line_content[2].strip()[1:-1], sep=sep).astype(np.int32)
                    cv2.fillPoly(line_mask, [np.stack([x_points, y_points], axis=1)], 255)

                else:
                    raise ValueError('line label format error')
            else:
                break

    mask = line_mask
    mask[mask == 255] = 1
    return mask


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

        image = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w, _ = image.shape

        mask = create_mask(img_h=img_h,img_w=img_w,label_path=mask_path,LINE_HEIGHT_RATIO=0.5)

        if self.transform:
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
            train_transform=None,
            val_transform=None,
            num_workers=4,
            pin_memory=True):

    train_ds = SegmentDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform= train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
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
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    return train_loader,val_loader