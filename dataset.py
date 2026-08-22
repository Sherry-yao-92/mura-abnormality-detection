from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import config
import torch

DATA_ROOT = Path("data")

def load_img_table(split, body_part=None):
    df = pd.read_csv(DATA_ROOT / "MURA-v1.1" / f"{split}_image_paths.csv", header = None, names=["path"])

    df["study"] = df["path"].str.rsplit("/", n=1).str[0] + "/"
    df["body_part"] = df["path"].str.split("/").str[2]
    df["label"] = df["study"].str.contains("_positive").astype(int)

    if body_part is not None:
        df = df[df["body_part"] == body_part].reset_index(drop=True)

    return df

def check_labels(split):
    truth = pd.read_csv(DATA_ROOT / "MURA-v1.1" / f"{split}_labeled_studies.csv", header = None, names=["study", "true_label"])
    images = load_img_table(split)
    merged = pd.merge(images, truth, on="study", how="left")
    assert not merged["true_label"].isna().any()
    assert (merged["label"]).equals(merged["true_label"])
    print(f"[{split}] {len(merged)} images, {merged['study'].nunique()} studies — labels verified")
class MuraDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = DATA_ROOT / row["path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, int(row["label"]), row["study"]

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(config.FLIP_P),
        transforms.RandomRotation(config.ROTATION_DEG),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    ])

if __name__ == "__main__":

    # Verify if label_img_table match with label_study_table
    print("--- label check ---")
    check_labels("valid")
    check_labels("train")

    print("\n--- dataset smoke test ---")
    wrist_valid = load_img_table("valid", "XR_WRIST")
    eval_ds  = MuraDataset(wrist_valid, transform=get_eval_transform())
    train_ds = MuraDataset(wrist_valid, transform=get_train_transform())
    print(len(eval_ds))
    img, label, study = eval_ds[0]
    print(img.shape, img.dtype, img.min().item(), img.max().item())
    print(label, study)

    print("\n--- augmentation check ---")
    eval_1, _, _ = eval_ds[0]
    eval_2, _, _ = eval_ds[0]
    print("eval  identical twice :", torch.equal(eval_1, eval_2))
    train_1, _, _ = train_ds[0]
    train_2, _, _ = train_ds[0]
    print("train identical twice :", torch.equal(train_1, train_2))
    
    print("\n--- wrist statistics ---")
    for split in ["train", "valid"]:
        wrist = load_img_table(split, "XR_WRIST")
        print(f"{split} wrist statistics:")
        print(f"images                          : {len(wrist)}") # number of wrist images
        print(f"studies                         : {wrist['study'].nunique()}") # number of non-repeating wrist studies
        print(f"positive rate per image         : {wrist['label'].mean():.3f}") 
        print(f"positive rate per study         : {wrist.groupby('study')['label'].first().mean():.3f}") # choose the first label of each study
        print("distribution of images per study:") # index = #images, value = #studies
        print(wrist["study"].value_counts().value_counts().sort_index().to_string()) 