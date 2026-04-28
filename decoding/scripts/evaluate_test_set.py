import json
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataloader import WatermarkDataset
from model import WatermarkClassifier
from utils import compute_metrics, print_metrics

def evaluate_test_set():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load best model
    checkpoint = torch.load('best_model.pth', map_location=device, weights_only=False)
    model = WatermarkClassifier(backbone='resnet50')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Val exact match at save time: {checkpoint['metrics']['exact_match_rate']:.4f}")
    
    # Load test set
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = WatermarkDataset(
        metadata_path='../encoding/data/metadata.json',
        image_dir='../encoding/data/images/',
        transform=transform
    )
    
    with open('splits.json', 'r') as f:
        splits = json.load(f)
    
    test_dataset = Subset(full_dataset, splits['test'])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = batch['bits'].to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.append(preds)
            all_targets.append(targets)
    
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    metrics = compute_metrics(all_preds, all_targets)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS (unseen during training)")
    print("="*60)
    print_metrics(metrics, prefix='Test ')
    print("="*60)
    
    # Compare to validation
    val_exact = checkpoint['metrics']['exact_match_rate']
    test_exact = metrics['exact_match_rate']
    diff = abs(val_exact - test_exact)
    
    print(f"\nComparison:")
    print(f"  Val exact match (epoch 28):  {val_exact:.4f}")
    print(f"  Test exact match:            {test_exact:.4f}")
    print(f"  Difference:                  {diff:.4f} ({diff/val_exact*100:.1f}%)")
    
    if test_exact >= val_exact * 0.95:
        print("\n✅ EXCELLENT: Test within 5% of validation")
        print("   → Model generalizes well, minimal overfitting")
    elif test_exact >= val_exact * 0.85:
        print("\n⚠️  MILD OVERFITTING: Test 5-15% worse than validation")
        print("   → Acceptable for this task, consider early stopping next time")
    else:
        print("\n❌ SEVERE OVERFITTING: Test >15% worse than validation")
        print("   → Model memorized training data, needs regularization")

if __name__ == '__main__':
    evaluate_test_set()
