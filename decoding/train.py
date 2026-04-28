import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from dataloader import WatermarkDataset
from model import WatermarkClassifier
from utils import compute_metrics, print_metrics

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc='Training'):
        images = batch['image'].to(device)
        targets = batch['bits'].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    for batch in tqdm(loader, desc='Evaluating'):
        images = batch['image'].to(device)
        targets = batch['bits'].to(device)
        
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        
        # Convert to binary predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        all_preds.append(preds)
        all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(loader)
    
    return metrics

def main():
    # Config - optimized for Mac testing
    config = {
        'batch_size': 4,  # Small for Mac
        'num_epochs': 3,  # Just test 3 epochs locally
        'lr': 1e-4,
        'backbone': 'resnet50',
        'image_size': 512,  # Smaller for Mac
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = WatermarkDataset(
        metadata_path='../encoding/data/metadata.json',
        image_dir='../encoding/data/images/',
        transform=transform
    )
    
    # Load splits
    with open('splits.json', 'r') as f:
        splits = json.load(f)
    
    train_dataset = Subset(full_dataset, splits['train'])
    val_dataset = Subset(full_dataset, splits['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=0)  # 0 for Mac compatibility
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=0)
    
    # Model
    model = WatermarkClassifier(backbone=config['backbone']).to(config['device'])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    best_exact_match = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, 
                                     optimizer, config['device'])
        val_metrics = evaluate(model, val_loader, criterion, config['device'])
        
        print(f"\nTrain loss: {train_loss:.4f}")
        print_metrics(val_metrics, prefix='Val ')
        
        # Save best model
        if val_metrics['exact_match_rate'] > best_exact_match:
            best_exact_match = val_metrics['exact_match_rate']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'metrics': val_metrics
            }, 'best_model.pth')
            print(f"\n✓ Saved best model (exact match: {best_exact_match:.4f})")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best exact match rate: {best_exact_match:.4f}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
