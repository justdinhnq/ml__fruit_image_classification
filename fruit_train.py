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

import numpy as np

import torch.nn.functional as F
import torchvision.transforms.functional as FT
import time

# Set resize_size and batch_size
resize_size = 300
batch_size = 45

# Define transformations for training data
transform_train = transforms.Compose([
    transforms.Resize((resize_size,resize_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
# Specify classes
classes = ('cherry','strawberry', 'tomato')

### Step 1: define 2 models
# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3*300*300, 3*300)
        self.layer2 = nn.Linear(3*300, 120)
        self.layer3 = nn.Linear(120, 84)
        self.layer4 = nn.Linear(84, 3) # 3 classes

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
    
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

### Step 2: prepare data
# Function to check if an image is noisy
def is_noisy(img_path):
    # Read the image using OpenCV
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the standard deviation of pixel values
    std_dev = np.std(gray_img)
    
    # Set a threshold
    threshold = 20

    # Return True if the image is noisy, False otherwise
    return std_dev < threshold

# Function to detect noisy images
def detect_noisy_images(dataset):
    noisy_image_count = 0
    noisy_images = []
    non_noisy_images = []

    for img_path, class_ in dataset.imgs:
        # Check if the image is noisy
        if is_noisy(img_path):
            noisy_image_count += 1
            noisy_images.append((img_path, class_))
        else:
            non_noisy_images.append((img_path, class_))

    print(f'Number of noisy images: {noisy_image_count}')
    
    return noisy_image_count, noisy_images, non_noisy_images

# Function to perform data augmentation
def get_more_by_augment():
    folders = ['./traindata/cherry', './traindata/strawberry', './traindata/tomato']

    for folder in folders:
        # Passing the path of the image directory
        p = Augmentor.Pipeline(folder)

        # Defining augmentation parameters and generating 5 samples
        p.flip_left_right(0.5)
        p.black_and_white(0.1)
        p.rotate(0.3, 10, 10)
        p.skew(0.4, 0.5)
        p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5)
        p.sample(300)
        
    source_folders = ['./traindata/cherry/output', './traindata/strawberry/output', './traindata/tomato/output']

    for index, src_folder in enumerate(source_folders):
        # List all files in the source directory
        files = os.listdir(src_folder)

        # Move each file to the destination directory
        for file in files:
            source_file_path = os.path.join(src_folder, file)
            destination_file_path = os.path.join(folders[index], file)
            shutil.move(source_file_path, destination_file_path)
            
        shutil.rmtree(src_folder)

# Function to split data into train and test loaders
def prepare_train_test_data(dataset):
    size = len(dataset)
    train_size = int(0.8 * size)
    test_size = size - train_size

    # split the dataset into train and test
    train, test = random_split(
        dataset, [train_size, test_size]
    )

    # DataLoader
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train, test, train_loader, test_loader

# Function to split data into train and test
def split_data():
    data_root = 'traindata'
    custom_dataset = ImageFolder(root=data_root, transform=transforms.ToTensor())

    noisy_image_count, noisy_images, non_noisy_images = detect_noisy_images(custom_dataset)

    custom_dataset.transform = transform_train
    custom_dataset.imgs = non_noisy_images

    train, test, train_loader, test_loader = prepare_train_test_data(custom_dataset)
    
    return train, test, train_loader, test_loader

### Step 3: running models
# Function to train and evaluate a model
def train_eval(model, train_loader, test_loader, loss_fn, optimizer, epochs = 10):
    # Create lists to store loss and accuracy for each epoch for both train and test sets
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    # Training and evaluation loop
    for epoch in range(epochs):
        start_time = time.time()  # Record the start time of the epoch
        
        # Training
        model.train()
        running_loss_train = 0.0
        correct_train = 0
        total_train = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()
            correct_train += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss_train = running_loss_train / len(train_loader)
        epoch_accuracy_train = 100 * correct_train / total_train
        train_loss_list.append(epoch_loss_train)
        train_accuracy_list.append(epoch_accuracy_train)

        # Evaluation on Test set
        model.eval()
        running_loss_test = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, labels = data
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                running_loss_test += loss.item()
                correct_test += (torch.argmax(outputs, dim=1) == labels).sum().item()
                total_test += labels.size(0)

        epoch_loss_test = running_loss_test / len(test_loader)
        epoch_accuracy_test = 100 * correct_test / total_test
        test_loss_list.append(epoch_loss_test)
        test_accuracy_list.append(epoch_accuracy_test)
        
        end_time = time.time()  # Record the end time of the epoch
        epoch_time = end_time - start_time  # Calculate the time taken for the epoch

        print(f"Epoch {epoch+1} - Time: {epoch_time:.2f} seconds - Train Loss: {epoch_loss_train:.4f}, Train Accuracy: {epoch_accuracy_train:.2f}, Test Loss: {epoch_loss_test:.4f}, Test Accuracy: {epoch_accuracy_test:.2f}")

    # Return training and testing loss and accuracy lists for verification
    return train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list

# Function to save a trained model
def save_model(model, PATH = './mlp.pth'):
    torch.save(model.state_dict(), PATH)

    
def main():
    ##### Phase 1: data preparation
    print('Phase 1: data preparation')
    get_more_by_augment()
    
    train, test, train_loader, test_loader = split_data()

    small_train, small_test, small_train_loader, small_test_loader = prepare_train_test_data(test_loader.dataset.dataset)

    ##### Phase 2: run MLP model
    print('Phase 2: run MLP model')
    ### MLP model
    mlp = MLP()
    
    learning_rate = 0.01
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp.parameters(), lr=learning_rate, momentum=0.9)
    
    print(f'MLP model: {mlp}')
    
    train_eval(mlp, small_train_loader, small_test_loader, loss_fn, optimizer, 10)
    
    
    
    ##### Phase 3: run CNN model
    print('Phase 3: run CNN model')
    ### CNN model
    model = SimpleCNN()
    
    # L2 Regularization (Weight Decay)
    optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    print(f'CNN model: {model}')
    
    train_eval(model, train_loader, test_loader, loss_fn, optimizer, 20)
    print('Save CNN model')
    save_model(model, PATH = 'model.pth')

if __name__ == "__main__":
    main()    
    
    