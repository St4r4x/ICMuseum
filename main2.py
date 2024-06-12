import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import os

class PaintingDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = int(self.labels[idx])
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)  # Assurez-vous que le label est un tenseur PyTorch
        return image, label

# Définition des transformations avec data augmentation pour l'ensemble d'entraînement
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Recadrage aléatoire de l'image
    transforms.RandomHorizontalFlip(),  # Retournement horizontal aléatoire
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Définition des transformations sans data augmentation pour l'ensemble de test
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Charger les données
image_dir = "./musée"
image_paths = [os.path.join(image_dir, image) for image in os.listdir(image_dir) if image.endswith('.jpg')]
# Extraire les labels à partir des noms de fichiers
labels = [os.path.splitext(os.path.basename(path))[0].split('.')[0] for path in image_paths]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2)
num_epochs = 50
batch_size = 4

# Utilisation des transformations lors de la création des datasets
train_dataset = PaintingDataset(train_image_paths, train_labels, transform=train_transform)
test_dataset = PaintingDataset(test_image_paths, test_labels, transform=test_transform)

# Création des DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Définir le modèle
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(set(labels)))
model = model.to(device)  #

# Définir la fonction de perte et l'optimiseur
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Entraîner le modèle
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Transférer les données et les labels sur le GPU si disponible
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Évaluer le modèle
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Sauvegarder le modèle
torch.save(model.state_dict(), 'model.pth')

# Fonction de prédiction
def predict(image_path):
    model = models.resnet50(pretrained=False)  # Pas besoin de poids pré-entraînés car nous chargeons notre propre modèle
    model.fc = torch.nn.Linear(model.fc.in_features, len(set(labels)))
    model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)  # Déplacer le modèle sur le périphérique
    model.eval()
    image = Image.open(image_path)
    image = test_transform(image).unsqueeze(0)
    image = image.to(device)  # Déplacer l'image sur le même périphérique que le modèle
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted