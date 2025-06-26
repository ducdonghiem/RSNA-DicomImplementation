import torch
import torchvision
import torch.nn as nn

class ModelFactory:
    """Factory class for creating different model architectures."""
    
    @staticmethod
    def create_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
        """
        Create a model based on the specified architecture.
        
        Args:
            model_name: Name of the model ('resnet50', 'vit_b_16', etc.)
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            
        Returns:
            PyTorch model
        """
        model_name = model_name.lower()
        
        if model_name == 'resnet50':
            if pretrained:
                model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
            else:
                model = torchvision.models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # slowest and need strong pre training
        elif model_name == 'vit':
            if pretrained:
                model = torchvision.models.vit_b_16(weights='IMAGENET1K_V1')
            else:
                model = torchvision.models.vit_b_16(weights=None)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
        # CNN, much fastter than resnet50, less overfitting risk
        elif model_name == 'efficientnet_b0':
            if pretrained:
                model = torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')
            else:
                model = torchvision.models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            
        else:
            raise ValueError(f"Model {model_name} not supported")
            
        return model
