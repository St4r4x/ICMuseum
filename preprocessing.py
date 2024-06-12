# preprocessing.py
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

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

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Recadrage aléatoire de l'image
    transforms.RandomHorizontalFlip(),  # Retournement horizontal aléatoire
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])