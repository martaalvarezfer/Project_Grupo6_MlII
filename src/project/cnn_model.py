"""
Module: model_utils.py
Description: Define la arquitectura CNN para clasificación de imágenes y 
provee funciones de entrenamiento, predicción, guardado, carga y visualización de la historia de entrenamiento.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb


# =============================================================================
#                           Clase CNN
# =============================================================================
class CNN(nn.Module):
    """Convolutional Neural Network model for image classification."""

    def __init__(self, base_model, num_classes, unfreezed_layers=0):
        """
        CNN model initializer.

        Args:
            base_model (nn.Module): Pre-trained model to use as the base.
            num_classes (int): Number of classes in the dataset.
            unfreezed_layers (int): Number of layers to unfreeze from the base model.
        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Freeze all layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last N layers (by module)
        if unfreezed_layers > 0:
            for layer in list(self.base_model.children())[-unfreezed_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Detect and adapt final layer based on model architecture
        if hasattr(self.base_model, 'classifier') and isinstance(self.base_model.classifier, nn.Sequential):
            in_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier = nn.Identity()

        elif hasattr(self.base_model, 'fc'):  # ResNet-like
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

        elif hasattr(self.base_model, 'heads'):  # Vision Transformer (ViT)
            in_features = self.base_model.heads.head.in_features
            self.base_model.heads.head = nn.Identity()

        else:
            raise ValueError("Modelo base no compatible: no se encuentra la capa de salida reconocible.")

        # Nueva capa completamente conectada (head)
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_model(self, train_loader, valid_loader, optimizer, criterion, epochs, nepochs_to_save=10):
        with TemporaryDirectory() as temp_dir:
            best_model_path = os.path.join(temp_dir, 'best_model.pt')
            best_accuracy = 0.0
            torch.save(self.state_dict(), best_model_path)

            history = {
                'train_loss': [],
                'train_accuracy': [],
                'valid_loss': [],
                'valid_accuracy': []
            }

            for epoch in range(epochs):
                self.train()
                train_loss = 0.0
                train_accuracy = 0.0
                for images, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_accuracy += (outputs.argmax(dim=1) == labels).sum().item()

                train_loss /= len(train_loader)
                train_accuracy /= len(train_loader.dataset)
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)
                print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

                self.eval()
                valid_loss = 0.0
                valid_accuracy = 0.0
                for images, labels in valid_loader:
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    valid_accuracy += (outputs.argmax(dim=1) == labels).sum().item()

                valid_loss /= len(valid_loader)
                valid_accuracy /= len(valid_loader.dataset)
                history['valid_loss'].append(valid_loss)
                history['valid_accuracy'].append(valid_accuracy)
                print(f'Epoch {epoch + 1}/{epochs} - Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": valid_loss,
                    "train_acc": train_accuracy,
                    "val_acc": valid_accuracy
                })

                if epoch % nepochs_to_save == 0:
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy
                        torch.save(self.state_dict(), best_model_path)

            torch.save(self.state_dict(), best_model_path)
            self.load_state_dict(torch.load(best_model_path))
            return history

    def predict(self, data_loader):
        self.eval()
        predicted_labels = []
        for images, _ in data_loader:
            outputs = self(images)
            predicted_labels.extend(outputs.argmax(dim=1).tolist())
        return predicted_labels

    def save_model(self, filename: str):
        os.makedirs(os.path.dirname('models/'), exist_ok=True)
        full_path = os.path.join('models', filename + '.pt')
        torch.save(self.state_dict(), full_path)

    @staticmethod
    def _plot_training(history):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['valid_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()


# =============================================================================
#                      Funciones Auxiliares Externas
# =============================================================================

def load_data(train_dir, valid_dir, batch_size, img_size):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = torchvision.datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, len(train_data.classes)


def load_model_weights(filename: str):
    full_path = os.path.join('models', filename + '.pt')
    state_dict = torch.load(full_path)
    return state_dict
