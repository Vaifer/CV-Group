import torch.nn as nn
import torchvision.models as models
import timm
from timm import create_model


class Stack_GoogleNet(nn.Module):
    def __init__(self,num_classes=6):
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
            nn.Linear(1024, num_classes)  # Multi-class output for 6 classes
        )

    def forward(self, x):
        x = self.googlenet(x)
        x = self.fc(x)
        return x


class Stack_Vit(nn.Module):
    def __init__(self,num_classes=6):
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
            nn.Linear(125, num_classes)  # Multi-class output for 6 classes
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x
class Stack_Resnet(nn.Module):
    def __init__(self, num_classes=6):  # Add num_classes as an argument
        super(Stack_Resnet, self).__init__()

        # Load a pre-trained ResNet18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Get the input features of the final fully connected layer (FC layer)
        num_ftrs = self.resnet.fc.in_features

        # Remove the final fully connected layer of ResNet18 and replace it
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        # Define the new fully connected layer for multi-class classification
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),     # First FC layer: num_ftrs -> 256
            nn.ReLU(),                    # Activation function
            nn.Dropout(0.3),              # Dropout to prevent overfitting
            nn.Linear(256, 128),           # Second FC layer: 256 -> 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),            # Third FC layer: 128 -> 64
            nn.ReLU(),
            nn.Linear(64, num_classes)     # Output layer: 64 -> num_classes (e.g., 6 for 6-class classification)
        )

    def forward(self, x):
        # Forward pass through the ResNet18 model without its final classification layer
        x = self.resnet(x)
        # Pass through the custom fully connected layers
        x = self.fc(x)
        return x

class Stack_Inception(nn.Module):
    def __init__(self,num_classes=6):
        super(Stack_Inception, self).__init__()

        # Load the pre-trained model (Inception V4) from HuggingFace hub using timm
        self.model = create_model(model_name="hf_hub:timm/inception_v4.tf_in1k", pretrained=True)

        # Get the number of input features for the last classifier layer
        in_features = self.model.get_classifier().in_features

        # If the model has a 'classifier' attribute, delete it
        if hasattr(self.model, 'classifier'):
            del self.model.classifier

        # Replace the final fully connected layer with a custom linear layer for 6-class classification
        # print(in_features,"zzzzzz")
        self.model.last_linear =  nn.Linear(in_features, num_classes)
        # self.model.head = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),  # 第一个全连接层
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.5),  # 增加 Dropout，尝试 0.5
            nn.Linear(128, num_classes),  # 最后一层输出
        )

    def forward(self, x):
        # Forward pass through the model
        x=self.model(x)
        return x