import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import config
from dataset import load_img_table, MuraDataset, get_eval_transform
from model import build_model
from train import validate

def to_study_level(probs, labels, studies):
    df = pd.DataFrame({"study":studies, "prob":probs, "label":labels})
    assert df.groupby("study")["label"].nunique().max() == 1

    study_df = df.groupby("study").agg(
        prob = ("prob", "mean"),
        label=("label","first"),
        n_images=("prob", "size") 
    ).reset_index()

    return study_df["prob"].values, study_df["label"].values, study_df

def main():
    device = torch.device("cuda")
    valid_table = load_img_table("valid", body_part=config.BODY_PART)
    valid_ds = MuraDataset(valid_table, transform=get_eval_transform())
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    model.load_state_dict(torch.load("checkpoints/best_auc.pt", map_location=device))

    criterion = nn.CrossEntropyLoss()
    val_loss, img_level_auc, probs, labels, studies = validate(model, valid_loader, criterion, device)
    study_level_probs, study_level_labels, study_level_df = to_study_level(probs, labels, studies)
    study_level_auc = roc_auc_score(study_level_labels, study_level_probs)

    print(f"cases of image-level:{study_level_df['n_images'].sum()}")
    print(f"cases of study-level:{len(study_level_df)}")
    print(f"image-level AUC:{img_level_auc:.4f}")
    print(f"study-level AUC:{study_level_auc:.4f}")
          
if __name__ == "__main__":
    main()
