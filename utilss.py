import torch
import torchvision
from data import SegmentDataset
from torch.utils.data import DataLoader
#from keras import backend as K
#from tensorflow import tf2
import cv2
import numpy as np
import statistics


def save_checkpoint(state, filename="my_checkpoint_30.pth.tar"):
    print("=>Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=>Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

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

def save_predictions_as_imgs(
    loader, model, folder="C:\\Users\\Administrator\\PycharmProjects\\SegmentationLine\\save_image", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")




def dice_coeff(loader,model,device):
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for data,target in loader:
            data = data.to(device=device)
            target = target.float().unsqueeze(1).to(device=device)
            pred = torch.sigmoid(model(data))
            pred = (pred > 0.5).float()
            smooth = 1e-8
            intersection = (pred * target).sum()
            dice_score += ((2. * intersection) + smooth) / (pred.sum() + target.sum() + smooth)

    print(f"Dice score:{dice_score/len(loader)}")

"""def check_accuracy(loader, model, device):
    num_correct = 0,
    num_pixels = 0,
    dice_score = 0,
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            #num_correct += (preds == y).sum()
            #num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / (
                (preds + y,).sum() + 1e-8
            )

    print(f"Dice score:{dice_score/len(loader)}")
    model.train()"""


"""def crop_table(img_h, img_w, label_path,LINE_HEIGHT_RATIO)
    table_mask = np.zeros((img_h, img_w))
    with open(label_path, 'r') as f:

        label_lines = f.readlines()
        # draw line mask
        for line in label_lines:
            line_content = line.strip().split(' ') if '[' not in line else ['[' + a for a in line.strip().split('[')]
            line_type = line_content[0]
            if 'table' in line_type:
                if len(line_content) == 5:  # type, x, y, w, h
                    x, y, w, h = [int(num) for num in line_content[1:]]
                    cv2.rectangle(table_mask, (x, y), (x + w, y + h), 255,
                                  line_median_height)  # line_median_height can be fixed to 3
                elif len(line_content) == 3:  # type, x_points, y_points
                    if ', ' in line_content[1].strip()[1:-1]:
                        sep = ', '
                    else:
                        sep = ' '
                    x_points, y_points = np.fromstring(line_content[1].strip()[1:-1], sep=sep).astype(
                        np.int32), np.fromstring(line_content[2].strip()[1:-1], sep=sep).astype(np.int32)
                    # print(x_points.shape[0], y_points)
                    if x_points.shape[0] <= 2:
                        continue
                    else:
                        cv2.polylines(table_mask, [np.stack([x_points, y_points], axis=1)], True, 255, line_median_height)
            else:
                break"""



