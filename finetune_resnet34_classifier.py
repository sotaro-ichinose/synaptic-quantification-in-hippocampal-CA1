import os
import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def select_training_directory():
    """Open a dialog to select the fine-tuning dataset directory."""
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select fine-tuning dataset directory")
    return directory


def create_dataloader(data_dir, batch_size=32, num_workers=2):
    """Create a dataloader for fine-tuning."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataset, dataloader


def load_model(model_path, device):
    """Load a pretrained model checkpoint for fine-tuning."""
    checkpoint = torch.load(model_path, map_location=device)

    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device), checkpoint


def fine_tune_model(model, dataloader, device, num_epochs, learning_rate):
    """Perform fine-tuning on the provided dataset."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {avg_loss:.6f}"
        )


def save_model(model, checkpoint, model_path):
    """Overwrite the existing model with fine-tuned weights."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": checkpoint["num_classes"],
            "class_names": checkpoint["class_names"],
        },
        model_path,
    )

    print(f"Fine-tuned model saved to: {model_path}")


def main():
    torch.multiprocessing.freeze_support()

    data_dir = select_training_directory()
    if not data_dir:
        print("No folder was selected. The program will exit.")
        return

    batch_size = 32
    learning_rate = 1e-5
    num_epochs = 5
    model_path = "./models/resnet34_trained.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset, dataloader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
    )

    model, checkpoint = load_model(model_path, device)

    fine_tune_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
    )

    save_model(model, checkpoint, model_path)


if __name__ == "__main__":
    main()