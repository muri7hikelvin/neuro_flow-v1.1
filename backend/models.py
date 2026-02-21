"""
Neural network models for distributed training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    """CNN for MNIST digit recognition"""
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class CIFAR_ResNet(nn.Module):
    """ResNet-inspired model for CIFAR-10"""
    def __init__(self):
        super(CIFAR_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual block 1
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.shortcut1 = nn.Conv2d(64, 128, 1)
        
        # Residual block 2
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.shortcut2 = nn.Conv2d(128, 256, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)
        
    def forward(self, x):
        # Initial conv
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual block 1
        residual = out
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn2(self.conv2(out))  # Second conv without activation
        out += self.shortcut1(residual)
        out = F.relu(out)
        
        # Residual block 2
        residual = out
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut2(residual)
        out = F.relu(out)
        
        # Global pooling and classification
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

def create_model(model_type: str, device: torch.device):
    """Factory function to create models"""
    if model_type == 'mnist_cnn':
        return MNIST_CNN().to(device)
    elif model_type == 'cifar_resnet':
        return CIFAR_ResNet().to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)