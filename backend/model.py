import torch.nn as nn
import timm

class DRViTModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Load pre-trained ViT
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes
        )
        
        # Access the classifier head (depends on timm model structure)
        # For most ViT models in timm, it's model.head
        n_features = self.model.head.in_features
        
        # Replace head with Dropout + Linear for better Generalization (Accuracy boost)
        self.model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
