import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

# Parameters
num_epochs = 10
batch_size = 8
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
train_dir = "./data/train"
test_dir = "./data/test"

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Map labels to integers
        unique_labels = sorted(self.img_labels['label'].unique())
        self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        self.img_labels['label'] = self.img_labels['label'].map(self.label_to_int)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Data Preprocessing
train_paths = [os.path.join(train_dir, image) for image in os.listdir(train_dir) if image.endswith('.jpg')]
test_paths = [os.path.join(test_dir, image) for image in os.listdir(test_dir) if image.endswith('.jpg')]

# Extract labels from filenames
train_labels = [os.path.splitext(os.path.basename(path))[0].split('_')[0] for path in train_paths]
test_labels = [os.path.splitext(os.path.basename(path))[0] for path in test_paths]

# Map labels to unique integers
unique_labels = sorted(set(train_labels + test_labels))
label_to_int = {label: i for i, label in enumerate(unique_labels)}

# Convert labels to integers
train_labels = [label_to_int[label] for label in train_labels]
test_labels = [label_to_int[label] for label in test_labels]

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform)
test_dataset = CustomImageDataset(test_paths, test_labels, transform=test_transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'Train dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

# Create model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(unique_labels))  # Replace the final layer to match the number of classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Train model function
def train_model(model, criterion, optimizer, dataloader, num_epochs, device):
    model = model.to(device)
    model.train()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_corrects.double() / len(dataloader.dataset)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

# Train the model
model = train_model(model, criterion, optimizer, train_dataloader, num_epochs=num_epochs, device=device)

# Save the model
torch.save(model.state_dict(), 'model.pth')

# Test model function
def test_model(model, dataloader, device):
    model = model.to(device)
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(dataloader.dataset)
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# Test the model
test_model(model, test_dataloader, device=device)
