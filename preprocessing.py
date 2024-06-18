# preprocessing.py
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os

class PaintingDataset(Dataset):
    def __init__(self, root_dir, unique_labels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.labels = [self.get_label(file) for file in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(int(label), dtype=torch.long)  # Assurez-vous que le label est un tenseur PyTorch
        return image, label

    def get_label(self, file):
        if '_' in file:
            label = file.split('_')[0]
        else:
            label = os.path.splitext(file)[0]
        return self.label_to_int[label]
        
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Recadrage aléatoire de l'image
    transforms.RandomHorizontalFlip(),  # Retournement horizontal aléatoire
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])