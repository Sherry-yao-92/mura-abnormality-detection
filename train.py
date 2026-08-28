import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path
import pandas as pd

import config
from dataset import load_img_table, MuraDataset, get_train_transform, get_eval_transform
from model import build_model

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss= 0.0
    n= 0

    for imgs, labels, studies in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

    return running_loss / n

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n = 0
    all_probs = []
    all_labels = []
    all_studies = []

    for imgs, labels, studies in tqdm(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs.float(), dim=1)[:,1]

        running_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_studies.extend(studies)

    avg_loss = running_loss / n
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, auc, all_probs, all_labels, all_studies
            
def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    train_table = load_img_table("train", config.BODY_PART)
    train_ds = MuraDataset(train_table, transform=get_train_transform())
    train_loader = DataLoader(train_ds, batch_size= config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    valid_table = load_img_table("valid", config.BODY_PART)
    valid_ds = MuraDataset(valid_table, transform=get_eval_transform())
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device =torch.device("cuda")
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.amp.GradScaler()

    Path("checkpoints").mkdir(exist_ok=True)
    best_auc = 0.0
    best_loss = float("inf")
    epochs_without_improve = 0
    history = []

    for epoch in range(1, config.MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, auc, probs, labels, studies = validate(model, valid_loader, criterion, device)
        print(f"epoch {epoch:3d} train {train_loss:.4f} val {val_loss:.4f} AUC {auc:.4f}")

        if auc > best_auc + config.MIN_DELTA:
            best_auc = auc
            epochs_without_improve = 0
            torch.save(model.state_dict(), "checkpoints/best_auc.pt")
            size_mb = Path("checkpoints/best_auc.pt").stat().st_size / 1024 ** 2
            print(f"saved best_auc.pt ({size_mb:.1f} MB)")

        else: 
            epochs_without_improve += 1

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_loss.pt")
            print(f"saved best_loss.pt")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "auc": auc})

        if epochs_without_improve >= config.PATIENCE:
            print(f"early stop at epoch {epoch}, best saved AUC {best_auc:.4f}")

            break
    pd.DataFrame(history).to_csv("history_224.csv", index=False)



if __name__ == "__main__":
    main()