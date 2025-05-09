# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import platform
import csv
import argparse
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

def train_model(device, learning_rate, batch_size, use_dropout, use_batchnorm, run_id, optimizer_type):
    num_epochs = 100
    patience = 5

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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = AlexNetCIFAR10(use_dropout=use_dropout, use_batchnorm=use_batchnorm).to(device)
    model = model.float()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

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
        print(f'[Run {run_id}] Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(trainloader):.4f} | Train Acc: {train_acc:.2f}%')

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
        print(f'[Run {run_id}] Validation Accuracy: {acc:.2f}%')

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            if not os.path.isdir('checkpoint'):
                os.makedirs('checkpoint')
            model_path = f'./checkpoint/{optimizer_type}_run_{run_id}_best.pth'
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'[Run {run_id}] Early stopping triggered.')
                break

    print(f'[Run {run_id}] Best Validation Accuracy: {best_acc:.2f}%')
    return best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'rmsprop'],
                        help='Optimizer to use: sgd | adam | rmsprop')
    parser.add_argument('--lr', type=float, help='Specific learning rate (optional)')
    parser.add_argument('--batch_size', type=int, help='Specific batch size (optional)')
    parser.add_argument('--dropout', type=str, choices=['true', 'false'], help='Use dropout (true/false)')
    parser.add_argument('--batchnorm', type=str, choices=['true', 'false'], help='Use batchnorm (true/false)')

    try:
        args = parser.parse_args()
    except SystemExit:
        print("\n[ERROR] Invalid arguments passed. Check for typos in flags or values.")
        parser.print_help()
        exit(1)

    optimizer_type = args.optimizer.lower()
    device = get_device()

    learning_rates = [args.lr] if args.lr is not None else [0.01, 0.001, 0.0001]
    batch_sizes = [args.batch_size] if args.batch_size is not None else [32, 64, 128]
    dropout_options = [args.dropout == 'true'] if args.dropout is not None else [True, False]
    batchnorm_options = [args.batchnorm == 'true'] if args.batchnorm is not None else [False, True]

    run_id = 1
    results = []

    # Prepare CSV file
    csv_filename = f'results_{optimizer_type}.csv'
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Optimizer', 'LR', 'Batch Size', 'Dropout', 'Batch Norm', 'Final Accuracy'])

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for dropout in dropout_options:
                for batchnorm in batchnorm_options:
                    print(f'\n=== Starting Run {run_id} with {optimizer_type.upper()} ===')
                    print(f'LR: {lr}, Batch Size: {batch_size}, Dropout: {dropout}, BatchNorm: {batchnorm}\n')

                    best_val_acc = train_model(device, lr, batch_size, dropout, batchnorm, run_id, optimizer_type)

                    with open(csv_filename, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([run_id, optimizer_type.upper(), lr, batch_size, dropout, batchnorm, f'{best_val_acc:.2f}%'])

                    results.append({
                        'Run': run_id,
                        'Learning Rate': lr,
                        'Batch Size': batch_size,
                        'Dropout': dropout,
                        'BatchNorm': batchnorm,
                        'Best Validation Accuracy': best_val_acc
                    })

                    run_id += 1

    print("\n=== All Runs Complete ===\n")
    for res in results:
        print(res)



if __name__ == '__main__':
    main()
