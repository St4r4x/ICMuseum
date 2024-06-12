import torch

def test_model(model, dataloader, device='cpu'):
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    with torch.no_grad():  # No need to track gradients for testing
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f} %')

    return accuracy