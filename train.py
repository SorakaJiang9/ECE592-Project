import os
import sys
import csv
import time
import datetime
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dataf
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_ssim
from torchvision.utils import save_image

from net.model import DD_UIE
from utils.utils import *
from utils.LAB import *
from utils.LCH import *
from utils.FDL import *

# ---------------------------------------------------------
# Device and multi-GPU setup
# ---------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
device_ids = [0, 1, 2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype_np = "float32"

# ---------------------------------------------------------
# Utility: save sample images from the test set
# ---------------------------------------------------------
def sample_images(batches_done, model, x_test, y_test):
    """
    Save a generated sample from the test set for qualitative inspection.
    """
    model.eval()
    with torch.no_grad():
        i = random.randrange(0, x_test.shape[0])
        real_A = x_test[i].unsqueeze(0).to(device)  # input
        real_B = y_test[i].unsqueeze(0).to(device)  # ground truth

        fake_B = model(real_A)

        imgx = fake_B.clamp(0.0, 1.0).data
        imgy = real_B.clamp(0.0, 1.0).data

        img_sample = torch.cat((imgx, imgy), dim=-2)  # stack along height
        os.makedirs("images/results", exist_ok=True)
        save_image(
            img_sample,
            f"images/results/{batches_done}.png",
            nrow=1,
            normalize=True,
        )
    model.train()


# ---------------------------------------------------------
# Data loading: train / test (input & GT)
# ---------------------------------------------------------
def load_image_folder(folder, size=(256, 256)):
    """
    Load all images from a folder, sorted by numeric filename.
    Returns a numpy array of shape (N, H, W, 3) in float32.
    """
    imgs = []
    path_list = os.listdir(folder)
    path_list.sort(key=lambda x: int(x.split(".")[0]))
    for name in path_list:
        path = os.path.join(folder, name)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        imgs.append(img)
    arr = np.array(imgs).astype(dtype_np)
    return arr


# Train input / GT
train_input_np = load_image_folder("./data/Train/input/")
train_gt_np = load_image_folder("./data/Train/GT/")

# Test input / GT
test_input_np = load_image_folder("./data/Test/input/")
test_gt_np = load_image_folder("./data/Test/GT/")

# Convert to torch tensors in [0, 1], shape (N, C, H, W)
X_train = torch.from_numpy(train_input_np).permute(0, 3, 1, 2) / 255.0
y_train = torch.from_numpy(train_gt_np).permute(0, 3, 1, 2) / 255.0
x_test = torch.from_numpy(test_input_np).permute(0, 3, 1, 2) / 255.0
Y_test = torch.from_numpy(test_gt_np).permute(0, 3, 1, 2) / 255.0

print("train input shape :", X_train.shape)
print("train output shape:", y_train.shape)
print("test input shape  :", x_test.shape)
print("test output shape :", Y_test.shape)

# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------
train_dataset = dataf.TensorDataset(X_train, y_train)
train_loader = dataf.DataLoader(
    train_dataset, batch_size=9, shuffle=True, num_workers=4
)

# ---------------------------------------------------------
# Model: Dual-Domain Underwater Enhancement Network (DD-UIE)
# ---------------------------------------------------------
model = DD_UIE(in_channels=3, channels=16, num_resblock=4, num_memblock=4)
model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)

# ---------------------------------------------------------
# Losses
# ---------------------------------------------------------
pixel_loss = nn.L1Loss(reduction="sum").to(device)
ssim_loss = pytorch_ssim.SSIM().to(device)
lab_loss = lab_Loss().to(device)
lch_loss = lch_Loss().to(device)
fdl_loss_fn = FDL(
    loss_weight=1.0,
    alpha=2.0,
    patch_factor=4,
    ave_spectrum=True,
    log_matrix=True,
    batch_matrix=True,
).to(device)

# ---------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------
LR = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.5, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.8)

# ---------------------------------------------------------
# (Optional) load pretrained checkpoint
# ---------------------------------------------------------
use_pretrain = False
if use_pretrain:
    start_epoch = 967
    ckpt_path = f"saved_models/DD_UIE_{start_epoch}.pth"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded pretrained checkpoint from epoch {start_epoch}.")
else:
    start_epoch = 0
    print("No pretrained model used, training will start from scratch.")

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
os.makedirs("saved_models", exist_ok=True)

f1 = open("DD_UIE_PSNR.csv", "w", encoding="utf-8")
csv_writer1 = csv.writer(f1)
f2 = open("DD_UIE_SSIM.csv", "w", encoding="utf-8")
csv_writer2 = csv.writer(f2)

checkpoint_interval = 5
epochs = start_epoch
n_epochs = 2000
sample_interval = 1000

best_psnr = 0.0
prev_time = time.time()

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
for epoch in range(epochs, n_epochs):
    epoch_psnr_list = []

    for i, batch in enumerate(train_loader):
        model.train()

        # Move data to device
        input_img = batch[0].to(device)  # (B, 3, H, W)
        gt_img = batch[1].to(device)

        # Forward
        optimizer.zero_grad()
        output = model(input_img)

        # Losses
        loss_rgb = pixel_loss(output, gt_img) / (gt_img.size(2) ** 2)
        loss_lab_val = (
            lab_loss(output, gt_img)
            + lab_loss(output, gt_img)
            + lab_loss(output, gt_img)
            + lab_loss(output, gt_img)
        ) / 4.0
        loss_lch_val = (
            lch_loss(output, gt_img)
            + lch_loss(output, gt_img)
            + lch_loss(output, gt_img)
            + lch_loss(output, gt_img)
        ) / 4.0
        loss_ssim_val = 1 - ssim_loss(output, gt_img)
        ssim_value = -(loss_ssim_val.item() - 1)
        loss_fdl_val = fdl_loss_fn(output, gt_img)

        # Combined objective (weights kept as in your original script)
        loss_final = (
            loss_ssim_val * 10.0
            + loss_rgb * 10.0
            + loss_lch_val
            + loss_lab_val * 0.0001
            + loss_fdl_val * 10000.0
        )

        # Backward and optimize
        loss_final.backward()
        optimizer.step()

        # Compute PSNR on the current batch
        out_train = torch.clamp(output.detach(), 0.0, 1.0)
        psnr_train = batch_PSNR(out_train, gt_img, 1.0)
        epoch_psnr_list.append(psnr_train)

        # Time and remaining estimate
        batches_done = epoch * len(train_loader) + i
        batches_left = n_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(
            seconds=batches_left * (time.time() - prev_time)
        )
        prev_time = time.time()

        # Logging to stdout
        if batches_done % 100 == 0:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [PSNR: %.4f] [SSIM: %.4f] "
                "[loss: %.4f] [loss_lch: %.4f] [loss_lab_scaled: %.6f] [fdl_loss_scaled: %.4f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_loader),
                    psnr_train,
                    ssim_value,
                    loss_final.item(),
                    loss_lch_val.item(),
                    loss_lab_val.item() * 0.0001,
                    loss_fdl_val.item() * 5000.0,
                    time_left,
                )
            )

        # Save qualitative samples and log PSNR/SSIM every sample_interval
        if batches_done % sample_interval == 0:
            sample_images(batches_done, model, x_test, Y_test)
            csv_writer1.writerow([str(psnr_train)])
            csv_writer2.writerow([str(ssim_value)])

    # Epoch-level PSNR
    PSNR_epoch = np.array(epoch_psnr_list)
    mean_psnr = PSNR_epoch.mean()

    # Save best checkpoint
    if mean_psnr > best_psnr:
        best_psnr = mean_psnr
        ckpt_path = f"saved_models/DD_UIE_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print("\nSaved checkpoint at epoch %d with PSNR = %.4f" % (epoch, best_psnr))

    scheduler.step()
