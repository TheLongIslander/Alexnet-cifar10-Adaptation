# model.py
import torch.nn as nn

class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10, use_dropout=True, use_batchnorm=False):
        super(AlexNetCIFAR10, self).__init__()
        
        def conv_block(in_channels, out_channels, kernel_size, stride, padding):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                      nn.ReLU(inplace=True)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            return layers
        
        self.features = nn.Sequential(
            *conv_block(3, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(64, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            *conv_block(192, 384, kernel_size=3, stride=1, padding=1),
            *conv_block(384, 256, kernel_size=3, stride=1, padding=1),
            *conv_block(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout() if use_dropout else nn.Identity(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x
