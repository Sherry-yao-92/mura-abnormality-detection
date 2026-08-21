from pathlib import Path
import pandas as pd

DATA_ROOT = Path("data")

def load_index(split, body_part=None):
    df = pd.read_csv(DATA_ROOT / "MURA-v1.1" / f"{split}_image_paths.csv", header = None, names=["path"])

    df["study"] = df["path"].str.rsplit("/", n=1).str[0] + "/"
    df["body_part"] = df["path"].str.split("/").str[2]
    df["label"] = df["study"].str.contains("_positive").astype(int)

    if body_part is not None:
        df = df[df["body_part"] == body_part].reset_index(drop=True)

    return df

def check_labels(split):
    truth = pd.read_csv(DATA_ROOT / "MURA-v1.1" / f"{split}_labeled_studies.csv", header = None, names=["study", "true_label"])
    images = load_index(split)
    merged = pd.merge(images, truth, on="study", how="left")
    assert not merged["true_label"].isna().any()
    assert (merged["label"]).equals(merged["true_label"])
    print(f"[{split}] {len(merged)} images, {merged['study'].nunique()} studies — labels verified")


if __name__ == "__main__":
    df = load_index("valid", "XR_WRIST")
    print(df.shape)
    print(df.head())
    check_labels("valid")
    check_labels("train")


