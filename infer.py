# infer.py
import torch
from torchvision import transforms,models
from PIL import Image
import os
import torch.nn as nn

# Définissez vos transformations d'inférence
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def load_model(model_path, num_classes, device):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # Change the number of output classes
    model.load_state_dict(torch.load(model_path))
    model.to(device)  # Move the model to the correct device
    return model


def infer(model, image_path, device):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Move the input data to the correct device

    with torch.no_grad():  # No need to track gradients for inference
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--images_dir', type=str, help='Path to the directory with images to infer')
    parser.add_argument('--model_path', type=str, help='Path to the saved model')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    num_classes = 20  # Change this to the number of classes you trained your model with
    model = load_model(args.model_path, num_classes, device)

    for filename in os.listdir(args.images_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(args.images_dir, filename)
            prediction = infer(model, image_path,device)
            print(f'Prediction for {filename}: {prediction}')