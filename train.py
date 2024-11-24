import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
from tqdm import tqdm

def train():
    # Force CPU usage
    device = torch.device("cpu")
    
    # Enhanced transforms for better generalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Increased batch size for better batch statistics
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # Changed optimizer to SGD with momentum and adjusted learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=1)
    
    # Train for 1 epoch
    model.train()
    pbar = tqdm(train_loader, desc='Training')
    running_loss = 0.0
    running_acc = 0.0
    total_batches = len(train_loader)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate batch accuracy for progress bar
        pred = output.argmax(dim=1, keepdim=True)
        acc = pred.eq(target.view_as(pred)).sum().item() / len(target)
        running_loss += loss.item()
        running_acc += acc
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.4f}',
            'avg_acc': f'{(running_acc / (batch_idx + 1)):.4f}'
        })
    
    # Print final metrics
    final_loss = running_loss / total_batches
    final_acc = running_acc / total_batches
    print(f"\nTraining completed:")
    print(f"Final average loss: {final_loss:.4f}")
    print(f"Final average accuracy: {final_acc:.4f}")
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_str = f"{final_acc:.4f}".replace(".", "p")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), 
              f'models/model_{timestamp}_acc{acc_str}.pth',
              _use_new_zipfile_serialization=False)
    
if __name__ == "__main__":
    train() 