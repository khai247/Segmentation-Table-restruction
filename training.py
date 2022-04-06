import torch
import albumentations as A
from albumentations.pytorch import toTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
"""from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)"""
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
train_img_dir =
train_mask_dir =
val_img_dir =
val_mask_dir =

def train(loader, model, optimizer,loss_fn,scaler):
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
    pass
if __name__ == "__main__":
    main()
