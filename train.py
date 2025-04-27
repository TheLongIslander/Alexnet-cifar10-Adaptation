# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import platform
from model import AlexNetCIFAR10

def get_device():
    if torch.backends.mps.is_available() and platform.machine() == "arm64":
        print("Using MPS backend on ARM64 Mac")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA backend")
        return torch.device("cuda")
    else:
        print("Using CPU backend")
        return torch.device("cpu")

def main():
    # Settings
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100
    optimizer_type = 'SGD'  # Choose 'SGD', 'Adam', 'RMSprop'
    use_dropout = True
    use_batchnorm = False
    patience = 5  # for early stopping

    device = get_device()

    # Data Preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)  # Changed here

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)  # Changed here

    # Model
    model = AlexNetCIFAR10(use_dropout=use_dropout, use_batchnorm=use_batchnorm).to(device)
    model = model.float()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        raise ValueError('Unsupported optimizer type')

    # Training Loop
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(trainloader):.4f} | Train Acc: {train_acc:.2f}%')

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        print(f'Validation Accuracy: {acc:.2f}%')

        # Save best model and reset patience
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(), './checkpoint/alexnet_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f'Best Validation Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
