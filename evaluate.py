import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score

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

def compute_metrics(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    accuracy = (tn+tp)/(tn+fp+fn+tp)
    sensitivity = tp/(fn+tp)
    specificity = tn/(tn+fp)
    precision = tp/ (fp+tp)
    f1 = 2*precision*sensitivity/(precision+sensitivity)

    return{
        "auc": roc_auc_score(labels, probs),
        "kappa": cohen_kappa_score(labels, preds),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":precision,
        "f1":f1,
        "tn":tn, "fp":fp, "fn":fn, "tp":tp
    }


def main():
    device = torch.device("cuda")
    valid_table = load_img_table("valid", body_part=config.BODY_PART)
    valid_ds = MuraDataset(valid_table, transform=get_eval_transform())
    valid_loader = DataLoader(valid_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    model.load_state_dict(torch.load("checkpoints/best_auc.pt", map_location=device))

    criterion = nn.CrossEntropyLoss()
    val_loss, _, probs, labels, studies = validate(model, valid_loader, criterion, device)
    study_probs, study_labels, study_df = to_study_level(probs, labels, studies)

    img_metrics   = compute_metrics(probs, labels)
    study_metrics = compute_metrics(study_probs, study_labels)

    comparison = pd.DataFrame({
        "image-level": img_metrics,
        "study-level": study_metrics,
    })
    print(comparison.round(4))
          
if __name__ == "__main__":
    main()
