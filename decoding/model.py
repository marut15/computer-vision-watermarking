import torch
import torch.nn as nn
from torchvision import models

class WatermarkClassifier(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        
        if backbone == 'resnet50':
            # Use weights parameter instead of pretrained
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            else:
                weights = None
            self.backbone = models.resnet50(weights=weights)
            
            # Remove final FC layer
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # 8 binary classifiers (sigmoid outputs)
        self.classifier = nn.Linear(num_features, 8)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits  # raw logits, apply sigmoid during inference

# Test
if __name__ == '__main__':
    model = WatermarkClassifier()
    x = torch.randn(2, 3, 512, 512)  # Smaller for Mac testing
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"✓ Model initialized successfully")
