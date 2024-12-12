import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vision_transformer import vit_b_16


# Define the Dataset class for keypoints
class KeypointAndBboxModel(nn.Module):
    def __init__(self, num_keypoints=50):
        super(KeypointAndBboxModel, self).__init__()

        # Backbone: Pretrained ResNet to extract features
        resnet = models.resnet34(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Exclude last FC layer and avg pool

        # Transformer block for capturing global context
        self.transformer = vit_b_16(pretrained=True)
        self.transformer.heads = nn.Identity()  # Remove transformer classification head

        # Feature fusion (concatenate CNN and Transformer features)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7 + 768, 1024),  # Adjust input size based on ResNet and Transformer output
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 4 + num_keypoints * 2),  # Output for bounding box and keypoints
            nn.Sigmoid()  # To keep output between 0 and 1 for normalization
        )

    def forward(self, x):
        # ResNet backbone forward pass
        resnet_features = self.resnet_backbone(x)

        # Vision Transformer forward pass
        transformer_features = self.transformer(x)

        # Concatenate CNN and Transformer features
        combined_features = torch.cat([resnet_features.flatten(1), transformer_features.flatten(1)], dim=1)

        # Fully connected layers to predict bounding box and keypoints
        output = self.fc(combined_features)

        return output

# Define the complex model architecture with Feature Pyramid Network (FPN) and Attention Mechanism
class EnhancedKeypointsModel(nn.Module):
    def __init__(self, num_keypoints=50):
        super(EnhancedKeypointsModel, self).__init__()
        resnet = models.resnet50(pretrained=True)

        # Backbone using ResNet layers
        self.resnet_backbone = nn.Sequential(
            *list(resnet.children())[:-2]  # Remove last FC and avg pool layers
        )

        # Feature Pyramid Network (FPN) for multi-scale feature extraction
        self.fpn = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Reduce to a 7x7 feature map
        )

        # Attention mechanism to focus on important regions
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()  # Output attention map between 0 and 1
        )

        # Fully connected layers for predicting keypoints
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_keypoints * 2),  # Predict 50 keypoints (x, y) pairs
            nn.Sigmoid()  # Output between 0 and 1 for normalized coordinates
        )

    def forward(self, x):
        # Extract features using ResNet backbone
        features = self.resnet_backbone(x)

        # Apply FPN for multi-scale features
        fpn_features = self.fpn(features)

        # Apply attention mechanism
        attention_map = self.attention(fpn_features)
        attended_features = fpn_features * attention_map  # Weight features with attention

        # Fully connected layers for keypoint prediction
        keypoints = self.fc(attended_features)
        return keypoints
