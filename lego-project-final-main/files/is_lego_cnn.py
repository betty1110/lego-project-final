from torch import nn

class IsLego(nn.Module):
    def __init__(self):
        super(IsLego, self).__init__()
        
        # Feature extractor: series of conv layers with ReLU and max pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # input: 3x64x64 -> output: 32x64x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x64x64 -> 32x32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x32x32 -> 64x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x32x32 -> 64x16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 64x16x16 -> 128x16x16
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x16x16 -> 128x8x8

            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x8x8 -> 128x8x8
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x8x8 -> 128x4x4
        )

        self.flatten = nn.Flatten()  # flattens to a vector
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # fully connected layer
            nn.ReLU(),
            nn.Linear(256, 1),  # output single value (e.g. binary classification)
        )
        
    def forward(self, x):
        x = self.features(x)  # extract features
        x = self.flatten(x)   # flatten tensor
        x = self.fc(x)        # pass through fully connected layers

        return x
