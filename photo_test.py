import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import pandas as pd
import pickle

# Charger le modèle pré-entraîné
model = models.resnet50(pretrained=False)
csv_file = 'test_labels.csv'
# Charger le mapping pendant le test
with open('label_to_int_mapping.pkl', 'rb') as f:
    label_to_int = pickle.load(f)
# Lire le fichier CSV
df = pd.read_csv(csv_file)
# Récupérer les labels uniques
unique_labels = df['label'].unique()

# Compter le nombre de classes
num_classes = len(unique_labels)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Définir les transformations pour l'image d'entrée
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Charger le mapping des labels
int_to_label = {i: label for label, i in label_to_int.items()}

def predict_image(image_path):
    # Charger et prétraiter l'image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Ajouter une dimension pour le batch

    # Passer l'image à travers le modèle pour obtenir les prédictions
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        predicted_label = int_to_label[predicted.item()]
        confidence = probabilities[0][predicted].item()

    return predicted_label, confidence
# Exemple d'utilisation
image_path = './IMG_0459.jpg'  # Remplacez par le chemin de votre image
predicted_class, confidence = predict_image(image_path)
print(f'L\'image appartient à la classe: {predicted_class} avec une confiance de {confidence*100:.2f}%')