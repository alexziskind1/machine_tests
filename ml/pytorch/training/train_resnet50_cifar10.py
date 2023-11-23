import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

batch_size = 128
num_epochs = 5
#num_samples = 1000  # Number of samples to use for a quick test

# Function to check and use GPU if available
def get_device():
    # Check for CUDA GPU
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU).")
        return torch.device("cuda")

    # Check for Apple Silicon GPU
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU).")
        return torch.device("mps")

    # Default to CPU
    else:
        print("Using CPU.")
        return torch.device("cpu")

# Select the device
device = get_device()

# Load CIFAR-100 dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resizing images to fit ResNet input size
    transforms.ToTensor(),
])
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
#train_subset = Subset(train_dataset, range(num_samples))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = models.resnet50(weights=None, num_classes=100).to(device)


# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"Batch {i + 1}, Loss: {loss.item()}")

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

print('Finished Training')
