import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("history_224.csv")

best_loss_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
best_auc_epoch = df.loc[df["auc"].idxmax(), "epoch"]

fig, axes = plt.subplots(1, 2, figsize = (11, 4))

axes[0].plot(df["epoch"], df["train_loss"], label = "train", color = "tab:blue")
axes[0].plot(df["epoch"], df["val_loss"], label = "valid", color = "tab:orange")
axes[0].axvline(best_loss_epoch, color = "gray", linestyle = "--", linewidth = 1)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("cross-entropy loss")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(df["epoch"], df["auc"], label = "valid AUC", color = "tab:orange")
axes[1].axvline(best_auc_epoch, color = "gray", linestyle = "--", linewidth = 1)
axes[1].set_xlabel("epoch")
axes[1].set_ylim(0.5, 1.0)
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("AUC")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("history_224.png", dpi=150)
print("saved history_224.png")