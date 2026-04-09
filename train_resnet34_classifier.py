import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


def select_dataset_directory():
    """Open a dialog to select the ImageFolder dataset directory."""
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select ImageFolder Dataset")
    return directory


def create_dataloaders(data_dir, batch_size=32, val_ratio=0.15, num_workers=2):
    """Create training and validation dataloaders."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = ImageFolder(data_dir, transform=transform)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataset, train_loader, val_loader


def build_model(num_classes, device):
    """Initialize the ResNet-34 model."""
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """Train the model and record training/validation losses."""
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Validation Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses


def save_model(model, class_names, save_dir):
    """Save the trained model and associated metadata."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "resnet34_trained.pth")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(class_names),
            "class_names": class_names,
        },
        save_path,
    )

    print(f"Model saved to: {save_path}")


def save_loss_curve(train_losses, val_losses, num_epochs, save_dir):
    """Save the training and validation loss curve as an EPS file."""
    plt.rcParams.update({"font.size": 24})

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="s")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_eps_path = os.path.join(save_dir, "loss_curve.eps")
    plt.savefig(loss_eps_path, format="eps")
    plt.close()

    print(f"Loss curve saved to: {loss_eps_path}")


def evaluate_model(model, val_loader, device):
    """Run inference on the validation set and return true/predicted labels."""
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.numpy())

    return y_true, y_pred


def save_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Generate and save the confusion matrix as an EPS file."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 24})
    sns.set(font_scale=1.5)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"shrink": 0.75},
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_eps_path = os.path.join(save_dir, "confusion_matrix.eps")
    plt.savefig(cm_eps_path, format="eps")
    plt.close()

    print(f"Confusion matrix saved to: {cm_eps_path}")


def main():
    data_dir = select_dataset_directory()
    if not data_dir:
        print("No folder was selected. The program will exit.")
        return

    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10
    val_ratio = 0.15
    save_dir = "./models"

    dataset, train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes=len(dataset.classes), device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
    )

    save_model(model, dataset.classes, save_dir)
    save_loss_curve(train_losses, val_losses, num_epochs, save_dir)

    y_true, y_pred = evaluate_model(model, val_loader, device)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=dataset.classes))

    save_confusion_matrix(y_true, y_pred, dataset.classes, save_dir)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()