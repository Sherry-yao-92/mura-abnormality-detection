import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import config

def build_model(num_classes=2):
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

if __name__ == "__main__":
    model = build_model()
    print(model.classifier)

    x = torch.randn(4, 3, config.IMG_SIZE, config.IMG_SIZE) #(input, channel, height, width)
    out = model(x)
    print(out.shape)