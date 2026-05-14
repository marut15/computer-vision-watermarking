import os
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def _data_root() -> Path:
    env = os.environ.get("PROJECT_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "data" / "computer-vision-watermarking"

class WatermarkDataset(Dataset):
    def __init__(self, metadata_path, image_dir, transform=None):
        """
        Args:
            metadata_path: path to metadata.json
            image_dir: path to encoding/data/images/
            transform: torchvision transforms
        """
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        entry = self.metadata[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, entry['file'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert bits to tensor
        bits = torch.tensor(entry['bits'], dtype=torch.float32)
        
        return {
            'image': image,
            'bits': bits,
            'id_int': entry['id_int'],
            'filename': entry['file']
        }

# Test the loader
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Smaller for Mac testing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dr = _data_root()
    dataset = WatermarkDataset(
        metadata_path=str(dr / "watermark_encoding/data/metadata.json"),
        image_dir=str(dr / "watermark_encoding/data/images/"),
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Bits: {sample['bits']}")
    print(f"ID: {sample['id_int']}")
