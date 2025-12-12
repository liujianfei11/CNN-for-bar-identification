import torch
import torch.nn as nn
from torchvision import models

#def build_model(num_classes=2):
#    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#    num_features = model.fc.in_features
#    model.fc = nn.Linear(num_features, num_classes)
#    return model

def build_model(num_classes=2):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features ##分类头在 model.classifier
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
