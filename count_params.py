import torch
from MV_model import MV_Model

# Initialize the model
num_classes = 1
pretrained = True
feature_dim = 768

model = MV_Model(
    num_classes=num_classes,
    pretrained=pretrained,
    feature_dim=feature_dim
)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# 9mil for MV model with EfficientNetB0 backbone
# 99 mil for MV model with ConvNeXt-V1 backbone