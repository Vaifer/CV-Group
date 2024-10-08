import torch.nn as nn
import torchvision.models as models

class Stack_GoogleNet(nn.Module):
    def __init__(self):
        super(Stack_GoogleNet, self).__init__()

        # Initialize GoogleNet
        self.googlenet = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        output_size = self.googlenet.fc.in_features

        # Remove the fc layer of GoogleNet
        self.googlenet.fc = nn.Identity()

        # Create a new fully connected layer for multi-class classification
        self.fc = nn.Sequential(
            nn.Linear(output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 6)  # Multi-class output for 6 classes
        )

    def forward(self, x):
        x = self.googlenet(x)
        x = self.fc(x)
        return x


class Stack_Vit(nn.Module):
    def __init__(self):
        super(Stack_Vit, self).__init__()

        # Initialize ViT
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        output_size = self.vit.heads.head.in_features

        # Remove the fc layer
        self.vit.heads = nn.Identity()

        # New fully connected layer for multi-class classification
        self.fc = nn.Sequential(
            nn.Linear(output_size, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(125, 6)  # Multi-class output for 6 classes
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x


class Stack_SwinTransformer(nn.Module):
    def __init__(self):
        super(Stack_SwinTransformer, self).__init__()

        # Initialize Swin Transformer
        self.swin_transformer = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        output_size = self.swin_transformer.head.in_features

        # Remove the fc layer
        self.swin_transformer.head = nn.Identity()

        # New fully connected layer for multi-class classification
        self.fc = nn.Sequential(
            nn.Linear(output_size, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(125, 6)  # Multi-class output for 6 classes
        )

    def forward(self, x):
        x = self.swin_transformer(x)  # Extract features from Swin Transformer
        x = self.fc(x)  # Apply the custom fully connected layer
        return x
