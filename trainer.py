import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet

from utilss import (
    load_checkpoint,
    save_checkpoint,
    dice_coeff,
    save_predictions_as_imgs,
)
from data import get_loader
#Hyperparameters
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 30
num_workers = 2
image_height = 160
image_width = 240
pin_memory = True
load_model = True
train_img_dir = "F:\\linhtinh\\Data\\segment\\train_image"
train_mask_dir = "F:\\linhtinh\\Data\segment\\train_mask"
val_img_dir = "F:\\linhtinh\\Data\\segment\\val_image"
val_mask_dir = "F:\\linhtinh\\Data\\segment\\val_mask"
test_mask_dir = "F:\\linhtinh\\Data\\segment\\test_mask"
test_img_dir = "F:\\linhtinh\\Data\\segment\\test_image"

def train(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_index, (data,targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        #forward

        predictions = model(data)
        loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        #update tqdm loop
        loop.set_postfix(loss=loss.item())



def main():
    train_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
    model = UNet(in_channels=3,out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loader(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers,
        pin_memory
    )
    if load_model:
        load_checkpoint(torch.load("my_checkpoint_30.pth.tar"),model)
    #dice_coeff(val_loader,model,device)
    """save_predictions_as_imgs(
        val_loader, model, folder="C:\\Users\\Administrator\\PycharmProjects\\SegmentationLine\\save_image_test",
        device=device
    )"""
    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, loss_fn)

        #save model
        checkpoint={
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        #check accuracy
        #check_dice_coef(val_loader, model, device=device)
        dice_coeff(val_loader, model, device)
        #print ex
    save_predictions_as_imgs(
        val_loader, model, folder="C:\\Users\\Administrator\\PycharmProjects\\SegmentationLine\\save_image",
        device=device
    )


if __name__ == "__main__":
    main()
