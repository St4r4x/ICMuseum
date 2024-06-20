import os
import shutil
from PIL import Image, ImageEnhance

# Dossier contenant les images originales
input_dir = './data'

# Dossiers pour sauvegarder les images originales et générées
test_dir = './data/test'
train_dir = './data/train'

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

        # Créer des variations de luminosité
        for i in range(1, 6):
            enhancer = ImageEnhance.Brightness(image)
            new_image = enhancer.enhance(i * 0.2)  # Modifier la luminosité
            new_image.save(os.path.join(train_dir, f"{filename.split('.')[0]}_brightness_{i}.jpg"))

        # Créer des variations de contraste
        for i in range(1, 6):
            enhancer = ImageEnhance.Contrast(image)
            new_image = enhancer.enhance(i * 0.2)  # Modifier le contraste
            new_image.save(os.path.join(train_dir, f"{filename.split('.')[0]}_contrast_{i}.jpg"))

        # Créer des variations de netteté
        for i in range(1, 6):
            enhancer = ImageEnhance.Sharpness(image)
            new_image = enhancer.enhance(i * 0.2)  # Modifier la netteté
            new_image.save(os.path.join(train_dir, f"{filename.split('.')[0]}_sharpness_{i}.jpg"))

        # Créer des variations de couleur
        for i in range(1, 6):
            enhancer = ImageEnhance.Color(image)
            new_image = enhancer.enhance(i * 0.2)  # Modifier la couleur
            new_image.save(os.path.join(train_dir, f"{filename.split('.')[0]}_color_{i}.jpg"))