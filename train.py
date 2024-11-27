import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.cuda.amp import GradScaler, autocast

from model import AHF_U_Net
from dataset import BrainTumorDataset

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, targets):
        #outputs = torch.clamp(outputs, min=-10.0, max=10.0)
        return self.bce(outputs, targets)

def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def calculate_metrics(outputs, targets, threshold=0.5):
    """Calculate metrics for binary segmentation"""
    # Apply sigmoid and threshold
    predictions = (torch.sigmoid(outputs) > threshold).float()
    
    # Calculate intersection and union
    intersection = (predictions * targets).sum(dim=(1,2,3))
    union = predictions.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    
    # IoU (Jaccard)
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice coefficient
    dice = (2 * intersection + 1e-6) / (predictions.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1e-6)
    
    # True Positive, False Positive, False Negative
    tp = (predictions * targets).sum(dim=(1,2,3))
    fp = predictions.sum(dim=(1,2,3)) - tp
    fn = targets.sum(dim=(1,2,3)) - tp
    
    # Precision, Recall, Accuracy
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    
    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }

# Kaggle dataset path
csv_file = '../input/brats2020-training-data/BraTS20 Training Metadata.csv'
dataset = BrainTumorDataset(csv_file)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=22, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=22, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AHF_U_Net().to(device)
model = nn.DataParallel(model)

model.apply(initialize_weights)

criterion = Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=4e-2)
steps_per_epoch = len(train_dataloader)
scheduler = OneCycleLR(
                optimizer,
                max_lr=4e-2,
                epochs=10,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.2,  
                anneal_strategy='cos',
                cycle_momentum=True,
                div_factor=25.0)

torch.autograd.set_detect_anomaly(True)
best_val_loss = 1e5
patience_counter = 0
train_losses, val_losses = [], []
epochs = 10
patience = 2

scaler = torch.amp.GradScaler('cuda')

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for images, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False):
        optimizer.zero_grad()
        images = images.permute(0,3,1,2).to(device)
        targets = targets.permute(0,3,1,2).to(device)
        
        try:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Check for NaN values
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}")
                continue
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        except Exception as e:
            print(f"Error in training: {e}")
            continue

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    num_val = 0
    epoch_val_metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}
    
    with torch.no_grad():
        for images, targets in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", leave=False):
            images = images.permute(0,3,1,2).to(device)
            targets = targets.permute(0,3,1,2).to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)

                if torch.isnan(loss):
                    continue
                else:
                    val_loss += loss.item()
                    num_val += 1

                    batch_metrics = calculate_metrics(outputs, targets)
                    for key in epoch_val_metrics:
                        epoch_val_metrics[key] += batch_metrics[key]
    
    val_loss /= num_val
    val_losses.append(val_loss)

    for key in epoch_val_metrics:
        epoch_val_metrics[key] /= num_val

    print(train_loss, val_loss)

    print(f"Val  - IoU: {epoch_val_metrics['iou']:.4f}, "
          f"Dice: {epoch_val_metrics['dice']:.4f}, Precision: {epoch_val_metrics['precision']:.4f}, "
          f"Recall: {epoch_val_metrics['recall']:.4f}, Accuracy: {epoch_val_metrics['accuracy']:.4f}")
    

    scheduler.step(val_loss)

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
