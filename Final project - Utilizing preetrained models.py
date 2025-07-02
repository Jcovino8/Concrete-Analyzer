# Final project - Utilizing rpetrained models

# These are the libraries will be used for this lab.
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
torch.manual_seed(0)

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os

# Create your own dataset object

class Dataset(Dataset):
    def __init__(self, transform=None, train=True):
        directory = "C:/Users/jackc/OneDrive - Villanova University/Documents/Desktop/IBM Course/(6) AI Capstone project with Deep Learning and Keras/data/concrete_data_week3/concrete_data_week3/train"
        positive = "Positive"
        negative = "Negative"

        positive_file_path = os.path.join(directory, positive)
        negative_file_path = os.path.join(directory, negative)

        positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".jpg")]
        negative_files = [os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files.sort()

        number_of_samples = len(positive_files) + len(negative_files)
        self.all_files = [None] * number_of_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files

        self.Y = torch.zeros(number_of_samples).long()
        self.Y[::2] = 1  # cracked
        self.Y[1::2] = 0  # not cracked

        split_point = int(0.8 * number_of_samples)  # 80% train, 20% validation

        if train:
            self.all_files = self.all_files[:split_point]
            self.Y = self.Y[:split_point]
        else:
            self.all_files = self.all_files[split_point:]
            self.Y = self.Y[split_point:]

        self.len = len(self.all_files)

        self.transform = transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        image = Image.open(image_path).convert('RGB')
        y = self.Y[idx]

        if self.transform:
            image = self.transform(image)

        return image, y
    
print("done")

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match ResNet input
    transforms.ToTensor(),          # Convert PIL → Tensor
    transforms.Normalize(           # Normalize for ResNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = Dataset(transform=transform, train=True)
validation_dataset = Dataset(transform=transform, train=False)
print("done")



# Question 1: Preapre a pre-trained model
# Step 1: Load the pre-trained model resnet18
model = models.resnet18(pretrained=True)

# Step 2: Set the parameter cannot be trained for the pre-trained model
for param in model.parameters():
    param.requires_grad = False

#Step 3
model.fc = nn.Linear(in_features=512, out_features=2)

print(model)


# Question 2: Train the model
# Step 1: Create the loss function
criterion = nn.CrossEntropyLoss()


# Step 2: Create a training loader and validation loader object, the batch size should have 100 samples each.
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

# Step 3: Use the following optimizer to minimize the loss
optimizer = torch.optim.Adam(
    [param for param in model.parameters() if param.requires_grad],
    lr=0.001
)



n_epochs = 1
loss_list = []
accuracy_list = []

N_test  = len(validation_dataset)
N_train = len(train_dataset)

start_time = time.time()

for epoch in range(n_epochs):
    # ---------- Training ----------
    model.train()                     # set to training mode
    epoch_loss = 0

    for x, y in train_loader:
        
        optimizer.zero_grad()         # clear gradients
        
        y_hat = model(x)              # forward pass (prediction)
        
        loss = criterion(y_hat, y)    # compute loss
        
        loss.backward()               # back-propagate
        
        optimizer.step()              # update parameters
        
        epoch_loss += loss.item()
        loss_list.append(loss.item())
    
    # ---------- Validation ----------
    model.eval()                      # set to evaluation mode
    correct = 0
    
    with torch.no_grad():             # no gradient calc on val
        for x_val, y_val in validation_loader:
            y_val_hat = model(x_val)                  # predict
            _, predicted = torch.max(y_val_hat, 1)    # class index
            correct += (predicted == y_val).sum().item()
    
    accuracy = correct / N_test
    accuracy_list.append(accuracy)
    
    print(f"Epoch {epoch+1} | "
          f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
          f"Val Accuracy: {accuracy:.4f}")

elapsed = (time.time() - start_time)/60
print(f"\nFinished 1 epoch in {elapsed:.2f} minutes.")
print(f"• Best validation accuracy this run: {max(accuracy_list):.4f}")

accuracy

plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()

# Question 3: Find misclassified images
misclassified = []
model.eval()

with torch.no_grad():
    for x_val, y_val in validation_loader:
        y_val_hat = model(x_val)
        _, predicted = torch.max(y_val_hat, 1)

        # Compare predictions with true labels
        for i in range(len(predicted)):
            if predicted[i] != y_val[i]:
                misclassified.append((x_val[i], predicted[i], y_val[i]))
                
            if len(misclassified) >= 4:
                break
        if len(misclassified) >= 4:
            break


