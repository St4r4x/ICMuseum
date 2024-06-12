# main.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import train_test_split
import os

# Importez vos modules
from preprocessing import PaintingDataset, train_transform
from train import train_model
from test import test_model

# Définissez vos paramètres
num_epochs = 100
batch_size = 8
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger les données
train_dir = "./train"
test_dir = "./test"

train_paths = [os.path.join(train_dir, image) for image in os.listdir(train_dir) if image.endswith('.jpg')]
test_paths = [os.path.join(test_dir, image) for image in os.listdir(test_dir) if image.endswith('.jpg')]

# Extraire les labels à partir des noms de fichiers
train_labels = [os.path.splitext(os.path.basename(path))[0][-1] for path in train_paths]
test_labels = [os.path.splitext(os.path.basename(path))[0][-1] for path in test_paths]

train_dataset = PaintingDataset(train_paths, train_labels, transform=train_transform)
test_dataset = PaintingDataset(test_paths, test_labels, transform=train_transform)
print(f'Train dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Créez votre modèle
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(set(train_labels)))  # Remplacez la dernière couche pour correspondre au nombre de classes

# Définissez votre critère et votre optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Entraînez votre modèle
model = train_model(model, criterion, optimizer, train_dataloader, num_epochs=num_epochs, device=device)

# Testez votre modèle
test_model(model, test_dataloader, device=device)