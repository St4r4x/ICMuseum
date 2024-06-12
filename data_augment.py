import os
import shutil
from PIL import Image
from torchvision import transforms

# Définition des transformations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Recadrage aléatoire de l'image
    transforms.RandomHorizontalFlip(),  # Retournement horizontal aléatoire
])

# Dossier contenant les images originales
input_dir = './musée'

# Dossiers pour sauvegarder les images originales et générées
test_dir = './test'
train_dir = './train'

# Créer les dossiers de sortie s'ils n'existent pas
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)

# Parcourir toutes les images dans le dossier d'entrée
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        # Déplacer l'image originale dans le dossier 'test'
        shutil.move(os.path.join(input_dir, filename), os.path.join(test_dir, filename))

        # Ouvrir l'image
        image = Image.open(os.path.join(test_dir, filename))
        
        # Générer des variantes de l'image
        for i in range(10):  # Générer 10 variantes pour chaque image
            # Appliquer les transformations
            augmented_image = transform(image)
            
            # Sauvegarder l'image générée dans le dossier 'train'
            augmented_image.save(os.path.join(train_dir, f'{os.path.splitext(filename)[0]}_augmented_{i}.jpg'))