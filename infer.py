# infer.py
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models

# Définissez vos transformations d'inférence
infer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_model(model_path):
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer_image(model, image_path, device='cpu'):
    image = Image.open(image_path)
    image = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():  # No need to track gradients for inference
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--image_path', type=str, help='Path to the image to infer')
    parser.add_argument('--model_path', type=str, help='Path to the saved model')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model_path)
    predicted_class = infer_image(model, args.image_path, device=device)

    print(f'The predicted class for the image {args.image_path} is {predicted_class}')