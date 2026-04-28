from torchvision import models
import torch.nn as nn

def get_model(num_classes):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model