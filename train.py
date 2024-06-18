import os
import torch
from tensorboardX import SummaryWriter

def train_model(model, criterion, optimizer, dataloader, num_epochs, device, num_classes):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Ensure CUDA operations are synchronous
    
    model = model.to(device)
    criterion = criterion.to(device)  # Ensure criterion is moved to the GPU if necessary
    model.train()  # Set the model to training mode
    writer = SummaryWriter()  # Initialize the TensorBoard writer

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Sanity checks
            assert inputs.is_cuda and labels.is_cuda, "Inputs and labels should be on the GPU"
            assert labels.min() >= 0, f"Label value is less than 0: {labels.min().item()}"
            assert labels.max() < num_classes, f"Label value exceeds number of classes: {labels.max().item()}"
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Log the loss
            if i % 10 == 0:  # Log every 10 batches
                writer.add_scalar('training loss', running_loss / ((i + 1) * inputs.size(0)), epoch * len(dataloader) + i)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

    writer.close()  # Close the TensorBoard writer
    return model