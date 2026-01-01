---
title: Dog Breed Classifier with CNN
category: Deep Learning
date: 2020-06-04
client: Udacity (Project)
description: Building a deep learning model that can identify dog breeds from images and detect whether an image contains a dog or human, using CNNs and transfer learning.
imageUrl: https://images.unsplash.com/photo-1587300003388-59208cc962cb?auto=format&fit=crop&q=80&w=800
link: "https://github.com/shrikantnaidu/Dog-Breed-Classifier"
tags:
  - CNN
  - Transfer Learning
  - PyTorch
  - Computer Vision
---

In this project, we develop an algorithm that could be used as part of a mobile or web app. The code accepts any user-supplied image as input—if a dog is detected, it provides an estimate of the dog's breed; if a human is detected, it provides an estimate of the dog breed that is most resembling.

![Sample Dog Output](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/sample_dog_output.png)

### The Road Ahead

We break the project into separate steps:

- **Step 0**: Import Datasets
- **Step 1**: Detect Humans
- **Step 2**: Detect Dogs
- **Step 3**: Create a CNN to Classify Dog Breeds from Scratch
- **Step 4**: Create a CNN to Classify Dog Breeds using Transfer Learning
- **Step 5**: Write our Algorithm
- **Step 6**: Test our Algorithm

### Step 0: Import Datasets

We start by loading the required human and dog datasets:

```python
import numpy as np
from glob import glob

# load filenames for human and dog images
human_files = np.array(glob("/data/lfw/*/*"))
dog_files = np.array(glob("/data/dog_images/*/*/*"))

print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))
```

**Output:**
```
There are 13233 total human images.
There are 8351 total dog images.
```

### Step 1: Detect Humans

We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.

```python
import cv2
import matplotlib.pyplot as plt

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
```

![Face Detection Example](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/output_3_1.png)

**Performance Assessment:**
- Percentage of Human Faces detected in human_files: **98%**
- Percentage of Human Faces detected in dog_files: **17%**

### Step 2: Detect Dogs

We use a pre-trained VGG-16 model to detect dogs in images. The model was trained on ImageNet, which contains over 10 million images across 1000 categories.

```python
import torch
import torchvision.models as models

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# dog detector using VGG16 predictions
def dog_detector(img_path):
    class_label = VGG16_predict(img_path)
    return ((class_label >= 151) & (class_label <= 268))
```

**Performance Assessment:**
- Dog Images in Human dataset: **1%**
- Dog Images in Dog Dataset: **100%**

### Step 3: Create a CNN from Scratch

We create a CNN that classifies dog breeds, aiming for at least 10% test accuracy.

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 500)
        self.fc2 = nn.Linear(500, 133)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**Training Results:**
```
Epoch: 25  Training Loss: 3.850253  Validation Loss: 3.939559
Test Accuracy: 11% (99/836)
```

### Step 4: Transfer Learning with VGG-16

We leverage transfer learning to significantly improve accuracy by using a pre-trained VGG-16 model and replacing only the final classifier layer.

```python
model_transfer = models.vgg16(pretrained=True)

# Freeze training for all "features" layers
for param in model_transfer.features.parameters():
    param.requires_grad = False

# Replace final layer for 133 dog breeds
n_inputs = model_transfer.classifier[6].in_features
model_transfer.classifier[6] = nn.Linear(n_inputs, 133)
```

**Training Results:**
```
Epoch: 15  Training Loss: 1.005084  Validation Loss: 0.391689
Test Accuracy: 86% (725/836)
```

### Step 5: The Algorithm

The final algorithm accepts an image and:
- If a **dog** is detected → returns the predicted breed
- If a **human** is detected → returns the resembling dog breed
- If **neither** is detected → returns an error

```python
def run_app(img_path):
    breed_pred = predict_breed_transfer(img_path)
    
    if dog_detector(img_path):
        print('Dog Detected')
        return print('The Predicted Breed:', breed_pred)
    
    elif face_detector(img_path):
        print('Human Detected')
        return print('Closest Dog Breed:', breed_pred)
    
    else:
        print('Error: Neither Dog nor Human Detected')
```

### Step 6: Test Results

![Human Detection Result](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/output_56_1.png)
*Human Detected → Closest Dog Breed: American Staffordshire Terrier*

![Dog Detection Result](https://raw.githubusercontent.com/shrikantnaidu/shrikantnaidu.github.io/main/_posts/assets/output_56_7.png)
*Dog Detected → The Predicted Breed: Bullmastiff*

### Conclusion

We successfully developed an algorithm that detects the dog breed for a dog image and estimates the closest resembling dog breed for a human. The model was initially built from scratch using CNNs, achieving 11% accuracy. By leveraging transfer learning with VGG-16, we dramatically improved performance to **86% accuracy**.

The complete implementation can be found in the [GitHub repository](https://github.com/shrikantnaidu/Dog-Breed-Classifier).
