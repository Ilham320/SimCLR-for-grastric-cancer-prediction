import torch
import torch.nn as nn
import torchvision.models as models

class SimCLR(nn.Module):
    """SimCLR model implementation"""
    
    def __init__(self, base_model='resnet50', out_dim=128, num_classes=8):
        super(SimCLR, self).__init__()
        
        # Create base encoder
        if base_model == 'resnet18':
            self.encoder = models.resnet18(weights=None)
            self.n_features = self.encoder.fc.in_features
        elif base_model == 'resnet34':
            self.encoder = models.resnet34(weights=None)
            self.n_features = self.encoder.fc.in_features
        elif base_model == 'resnet50':
            self.encoder = models.resnet50(weights=None)
            self.n_features = self.encoder.fc.in_features
        else:
            raise ValueError(f"Base model {base_model} not supported")
        
        # Replace the fc layer with an identity function
        self.encoder.fc = nn.Identity()
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, out_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(self.n_features, num_classes)
    
    def forward(self, x):
        # Get representations
        h = self.encoder(x)
        
        # Classification head
        return self.classifier(h)
    
    def forward_features(self, x):
        # Get representations
        h = self.encoder(x)
        
        # Project features
        z = self.projector(h)
        
        return h, z