import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils import get_dataloaders

MODEL = "resnet"   # change: resnet / mobilenet / efficientnet

def get_model(num_classes):
    if MODEL == "resnet":
        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif MODEL == "mobilenet":
        model = models.mobilenet_v2(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    elif MODEL == "efficientnet":
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    return correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _, num_classes = get_dataloaders("data_split")

    model = get_model(num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(10):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print("Training Done!")


if __name__ == "__main__":
    train()