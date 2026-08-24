import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
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

def main():
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

    loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
    print("epoch 1 train loss", loss)


if __name__ == "__main__":
    main()