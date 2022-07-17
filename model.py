import torch

import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model

__all__ = [
    'Alexnet_NI',
    'Alexnet_DD'
]


# Alexnet architecture as described in:
# https://www.analyticsvidhya.com/blog/2021/03/introduction-to-the-architecture-of-alexnet/
class Alexnet(nn.Module):
    def __init__(self,  n_classes):
        super().__init__()

        # Convolutional part
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))
        self.maxpooling1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.maxpooling2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.maxpooling3 = nn.MaxPool2d(3, 2)

        # MLP part
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 4096)
        self.drop2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, n_classes)

    # forward through convolutional part
    def conv_forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpooling1(x)

        x = F.relu(self.conv2(x))
        x = self.maxpooling2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpooling3(x)

        # flattens feature maps so the MLP can process them
        x = torch.flatten(x, start_dim=1)

        return x

    # forward through MLP
    def mlp_forward(self, x):
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x

    # forward method
    def forward(self, x):
        x = self.conv_forward(x)
        x = self.mlp_forward(x)
        return x


# Alexnet with 10 class head for natural images dataset
@register_model
def Alexnet_NI(pretrained=False, **kwargs):
    return Alexnet(8)


#  Alexnet with 2 class head for natural image dataset
@register_model
def Alexnet_DD(pretrained=False, **kwargs):
    return Alexnet(2)


# Testing Alexnet works correctly, with 5 random images and 10 classes
if __name__ == '__main__':
    from torchsummary import summary

    model = Alexnet(10, 3)

    summary(model, (3, 227, 227))

    # creating 5 images of the size alexnet wants and reshaping them channels, height, width
    x = torch.randn(5, 227, 227, 3).view(-1, 3, 227, 227)

    print(model(x).shape)
