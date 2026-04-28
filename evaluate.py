import torch
from torchvision import models
import torch.nn as nn
from utils import get_dataloaders

MODEL = "resnet"

def get_model(num_classes):
    if MODEL == "resnet":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif MODEL == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    elif MODEL == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, num_classes = get_dataloaders("data_split")

    model = get_model(num_classes)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    print(f"Test Accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    evaluate()