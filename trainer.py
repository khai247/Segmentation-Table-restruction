import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loader,
    check_accuracy,
    save_predictions_as_imgs,
)
#Hyperparameters
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
num_epochs = 100
num_workers =2
image_height = 160
image_width = 240
pin_memory = True
load_model = True
train_img_dir = "C:\\Users\\Administrator\\PycharmProjects\\SegmentationLine\\train_image"
train_mask_dir = "C:\\Users\\Administrator\\PycharmProjects\\SegmentationLine\\train_mask"
val_img_dir = "C:\\Users\\Administrator\\PycharmProjects\\SegmentationLine\\val_image"
val_mask_dir = "C:\\Users\\Administrator\\PycharmProjects\\SegmentationLine\\val_mask"

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_index, (data,targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.float().unsqueeze(1).to(device=device)

        #forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            ),
            ToTensorV2(),
        ]
    )
    model = UNet(in_channels=3,out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

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

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, loss_fn, scaler)

        #save model
        checkpoint={
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        #check accuracy
        check_accuracy(val_loader,model,device=device)
        #print ex
        save_predictions_as_imgs(
            val_loader,model,folder="saved_image/",device=device
        )


if __name__ == "__main__":
    main()
