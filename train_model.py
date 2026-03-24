import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import preprocessing_image_and_mask
from data_augmentation import augment_data
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
import numpy as np

from unet_model import UNet2D
from metrics import calculate_dice, calculate_iou, calculate_precision_recall

PLOTS_DIR = "Plots"
MODEL_DIR = "Models"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100


class PetDataset(Dataset):
    def __init__(self, root_dir, target_shape=(256, 256), augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.target_shape = target_shape
        self.augment = augment

        self.pet_folders = []
        for d in sorted(os.listdir(root_dir)):
            pet_path = os.path.join(root_dir, d)
            if not os.path.isdir(pet_path):
                continue

            image_path = os.path.join(pet_path, "image.jpg")
            mask_path = os.path.join(pet_path, "mask.png")

            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.pet_folders.append(d)
            else:
                print(f"Skipped pet {d}: missing image or mask")

        print(f"Found {len(self.pet_folders)} pets in {root_dir}")

    def __len__(self):
        return len(self.pet_folders)

    def __getitem__(self, index):
        pet_folder = self.pet_folders[index]
        pet_path = os.path.join(self.root_dir, pet_folder)

        image_path = os.path.join(pet_path, "image.jpg")
        mask_path = os.path.join(pet_path, "mask.png")

        image, mask = preprocessing_image_and_mask(image_path, mask_path)

        if self.augment:
            image, mask = augment_data(image, mask)

        transform = A.Compose([
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]),
                ToTensorV2()])
        result = transform(image=image, mask=mask)
        image, mask = result['image'], result['mask'].long()

        return image, mask


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        outputs_class1 = outputs[:, 1:2, :, :]
        masks_class1 = (masks == 1).float().unsqueeze(1)

        loss = criterion(outputs_class1, masks_class1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        dice = calculate_dice(outputs_class1, masks_class1)
        iou = calculate_iou(outputs_class1, masks_class1)
        precision, recall = calculate_precision_recall(outputs_class1, masks_class1)

        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)

    avg_loss = running_loss / len(loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    return avg_loss, avg_dice, avg_iou, avg_precision, avg_recall


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            outputs_class1 = outputs[:, 1:2, :, :]
            masks_class1 = (masks == 1).float().unsqueeze(1)

            loss = criterion(outputs_class1, masks_class1)
            running_loss += loss.item()

            dice = calculate_dice(outputs_class1, masks_class1)
            iou = calculate_iou(outputs_class1, masks_class1)
            precision, recall = calculate_precision_recall(outputs_class1, masks_class1)

            dice_scores.append(dice)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)

    avg_loss = running_loss / len(loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    return avg_loss, avg_dice, avg_iou, avg_precision, avg_recall


def test_model(model, test_loader, device):
    model.eval()
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []

    print("\nTesting best model")

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            outputs_class1 = outputs[:, 1:2, :, :]
            masks_class1 = (masks == 1).float().unsqueeze(1)

            dice = calculate_dice(outputs_class1, masks_class1)
            iou = calculate_iou(outputs_class1, masks_class1)
            precision, recall = calculate_precision_recall(outputs_class1, masks_class1)

            dice_scores.append(dice)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)

    print(f"\nTest Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Test IoU: {np.mean(iou_scores):.4f}")
    print(f"Test Precision: {np.mean(precision_scores):.4f}")
    print(f"Test Recall: {np.mean(recall_scores):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = PetDataset("Train", augment=True)
    val_dataset = PetDataset("Validation", augment=False)
    test_dataset = PetDataset("Test", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = UNet2D(in_channels=3, out_channels=2).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
    last_model_path = os.path.join(MODEL_DIR, "last_model.pth")

    best_val_dice = 0.0
    early_stop_patience = 15
    patience_counter = 0

    print("\nTraining started")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_dice, train_iou, train_precision, train_recall = train_epoch(
            model, train_loader, criterion, optimizer, device)

        val_loss, val_dice, val_iou, val_precision, val_recall = validate_epoch(
            model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

        scheduler.step(val_dice)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Dice: {best_val_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping on epoch: {epoch + 1}")
            break
        print()

    torch.save(model.state_dict(), last_model_path)
    print("Last model saved!\n")

    model.load_state_dict(torch.load(best_model_path))
    test_model(model, test_loader, device)