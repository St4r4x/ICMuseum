import torch.nn as nn
from torchvision import models

def create_model(model_type, num_classes):
    if model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_type == 'vgg16':
        model = models.vgg16(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    # Ajoutez d'autres types de modèles ici si nécessaire
    else:
        raise ValueError(f"Model type '{model_type}' not recognized.")
    
    return model