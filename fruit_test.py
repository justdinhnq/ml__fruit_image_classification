# link to my model: https://myvuwac-my.sharepoint.com/:f:/g/personal/nguyendinh1_myvuw_ac_nz/EtkOJLgxbs9JppBjTCgyZQIBDGyBJh2L-CNHtO4CpklvqA?e=iM3rad


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from skimage import color, feature, exposure

import random
import cv2

import shutil
import Augmentor
import os

from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F
import torchvision.transforms.functional as FT
import time

# Set resize_size and batch_size
resize_size = 300
batch_size = 45

# Define transformations for training data
transform = transforms.Compose([
    transforms.Resize((resize_size,resize_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
# Specify classes
classes = ('cherry','strawberry', 'tomato')


data_root = 'testdata'

def load_dataset():
    custom_dataset = ImageFolder(root=data_root, transform=transform)
    test_loader = DataLoader(custom_dataset, batch_size, num_workers=4, pin_memory=True)
    return test_loader

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # channel: 3 -> 6, kernel size = 5
        self.conv1 = nn.Conv2d(3,6,5)
        # 1/2 image size
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 45x82944 and 9*96*96x900
        self.fc1 = nn.Linear(9*96*96, 3*300)
        self.fc2 = nn.Linear(3*300, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 3) # 3 classes

    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv1(x))) 
        x = self.maxpool2(F.relu(self.conv2(x))) 
        
        # except batch, flatten all dimensions
        x = torch.flatten(x,1) 
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x    


def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # running images through the network
            outputs = model(images)
            # getting the highest energy class
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'[Test Images]Accuracy of the network: {100 * correct // total} %')

    # Prepare variables to store predictions
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # Get class's predictions
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # Showing accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.2f} %')

def main():
    """Main function includes 3 steps: load data, load model, evaluate model"""
    # Step 1: load test data
    print('Step 1: load test data')
    valid_loader = load_dataset()
    
    # Step 2: load model
    print('Step 2: load model')
    model = SimpleCNN()
    model.load_state_dict(torch.load('model.pth'))
    
    # Step 3: evaluate the model
    print('Step 3: evaluate the model')
    evaluate_model(model, valid_loader)

if __name__ == "__main__":
    main()
