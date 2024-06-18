# main.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import  transforms

# Importez vos modules
from preprocessing import PaintingDataset, train_transform
from model import create_model
from train import train_model
from test import test_model
from infer import infer

# Définissez vos paramètres
num_epochs = 10
batch_size = 8
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Créez votre modèle
model = create_model('resnet50', num_classes=20)

# Définissez votre fonction de perte et votre optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Créez vos chargeurs de données
train_dataset = PaintingDataset('./data/train', transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = PaintingDataset('./data/test', transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Entraînez votre modèle
model = train_model(model, criterion, optimizer, train_dataloader, num_epochs=num_epochs, device=device)

# Testez votre modèle
test_model(model, test_dataloader, device=device)