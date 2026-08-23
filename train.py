import torch
from torch.utils.data import DataLoader
import config
from dataset import load_img_table, MuraDataset, get_train_transform, get_eval_transform
from model import build_model

train_table = load_img_table("train", config.BODY_PART)
train_ds = MuraDataset(train_table, transform=get_train_transform())

train_loader = DataLoader(train_ds, batch_size= config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

valid_table = load_img_table("valid", config.BODY_PART)
valid_ds = MuraDataset(valid_table, transform=get_eval_transform())

valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

if __name__ == "__main__":
    imgs, labels, studies = next(iter(train_loader))
    print(imgs.shape, imgs.dtype)
    print(labels.shape, labels[:8])
    print(len(studies), studies[0])