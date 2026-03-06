import cv2
import torch
from torchvision import transforms

# Returns standard test transforms for images
def get_test_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),  # convert NumPy array to PIL image
        transforms.Resize((64, 64)),  # resize to 64x64
        transforms.ToTensor(),  # convert to tensor and scale to [0,1]
        transforms.Normalize(  # normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

# Resizing and conversion of image to tensor
def resizer(img):
    resized = cv2.resize(img, (64,64), interpolation=cv2.INTER_LINEAR)  # resize image

    resized = resized.astype('float32') / 255  # normalize to [0,1]

    resized = resized.transpose((2,0,1))  # change shape to CxHxW

    tensor = torch.from_numpy(resized)  # convert to tensor
    return tensor
