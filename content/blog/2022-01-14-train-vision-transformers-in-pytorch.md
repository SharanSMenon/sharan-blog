---
title: Train Vision Transformers in PyTorch
date: 2022-01-14T22:40:43.474Z
description: >-
  This tutorial trains Vision Transformers (DeIT) on a dataset of 50 butterfly
  species
---
Vision Transformers are a new type of image classification models that does away with convolutional layers while still achieving state-of-the-art results on image recognition benchmarks (like ImageNet). It also uses fewer resources than CNN's to train.

This article will focus on training a Vision Transformer model, known as [DeIT](https://github.com/facebookresearch/deit), on a dataset of [50 butterfly species](https://www.kaggle.com/gpiosenka/butterfly-images40-species) (the dataset had 50 butterfly species at the time of writing, the author updates this dataset frequently)

Full code is linked at the bottom of this tutorial.

## Libraries

For this project, you will need the following libraries

* `numpy` - `pip install numpy` (NumPy)
* `torch` - `pip install torch` (PyTorch)
* `timm` - `pip install timm` (Torchvision Image Models)
* `torchvision` - `pip install torchvision` (Torchvision)

```python
import numpy as np # linear algebra
import os
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T # for simplifying the transforms
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
## TIMM ## pip install timm
import timm
from timm.loss import LabelSmoothingCrossEntropy
import sys
from tqdm import tqdm
import time
import copy
```

### Removing Warnings (optional)

PyTorch and the other libraries might give a bunch of warnings, which can get annoying, use the following code to remove the warnings. 

```python
import warnings
warnings.filterwarnings("ignore")
```

## Load data

Now, time to load the butterfly dataset

```python
def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes
```

The above code will get a list of classes in your dataset.

```python
def get_data_loaders(data_dir, batch_size, train = False):
    if train:
        #train
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # imagenet means
            T.RandomErasing(p=0.2, value='random')
        ])
        train_data = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform = transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, len(train_data)
    else:
        # val/test
        transform = T.Compose([ # We dont need augmentation for test transforms
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # imagenet means
        ])
        val_data = datasets.ImageFolder(os.path.join(data_dir, "valid/"), transform=transform)
        test_data = datasets.ImageFolder(os.path.join(data_dir, "test/"), transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return val_loader, test_loader, len(val_data), len(test_data)
```

This function will create train, test, and validation data loaders, and apply augmentations on the training set. It resizes every image to 224x224 and applies ImageNet normalization on each image. 

```python
## I ran this code in a Kaggle Kernel
dataset_path = "/kaggle/input/butterfly-images40-species/butterflies/"
(train_loader, train_data_len) = get_data_loaders(dataset_path, 128, train=True)
(val_loader, test_loader, valid_data_len, test_data_len) = get_data_loaders(dataset_path, 32, train=False)
```

This just puts all the data loaders and dataset sizes in a easy to access place, these dictionaries will be used in the training function. 

```python
dataloaders = {
    "train": train_loader,
    "val": val_loader
}
dataset_sizes = {
    "train": train_data_len,
    "val": valid_data_len
}
```

The following 2 cells just ensure that the functions work as expected. Your code should produce output similar to what's shown below.

```python
classes = get_classes("/kaggle/input/butterfly-images40-species/butterflies/train/")
print(classes, len(classes))
```

```
[OUTPUT]
['adonis', 'american snoot', 'an 88', 'banded peacock', 'beckers white', 'black hairstreak', 'cabbage white', 'chestnut', 'clodius parnassian', 'clouded sulphur', 'copper tail', 'crecent', 'crimson patch', 'eastern coma', 'gold banded', 'great eggfly', 'grey hairstreak', 'indra swallow', 'julia', 'large marble', 'malachite', 'mangrove skipper', 'metalmark', 'monarch', 'morning cloak', 'orange oakleaf', 'orange tip', 'orchard swallow', 'painted lady', 'paper kite', 'peacock', 'pine white', 'pipevine swallow', 'purple hairstreak', 'question mark', 'red admiral', 'red spotted purple', 'scarce swallow', 'silver spot skipper', 'sixspot burnet', 'skipper', 'sootywing', 'southern dogface', 'straited queen', 'two barred flasher', 'ulyses', 'viceroy', 'wood satyr', 'yellow swallow tail', 'zebra long wing'] 50
```

```python
print(len(train_loader), len(val_loader), len(test_loader))
print(train_data_len, valid_data_len, test_data_len)
```

```
[OUTPUT]
39 8 8
4955 250 250
```

## Define Model

We will start by defining the device in which we are training on

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
```

```
[OUTPUT]
device(type='cuda')
```

It is recommended that you train this on a GPU, Kaggle and Google Colab both give you free GPU's that are more than enough for this simple task.

Load the model from `torch.hub`. It will take a second or 2 to load the model's pretrained weights, depending on your internet connection. 

```python
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
```

### Configuring the model

To train it on the butterfly dataset, the model needs to be changed. That means freezing the model and then changing the last layer so that we can train it on the butterfly dataset

```python
for param in model.parameters(): #freeze model
    param.requires_grad = False

n_inputs = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes)) # 50 classes
)
model = model.to(device)
```

The model was trained on ImageNet, which had 1000 classes, but the butterfly dataset has only 50 classes, so the output layer needs to be changed to reflect the new dataset. 

## Loss, Optimizer, and Hyperparameters

Define the loss function and the optimizer. Note that `LabelSmoothingCrossEntropy` is being used here instead of `nn.CrossEntropyLoss`. Label smoothing has been shown to get better results over normal cross entropy. Our optimizer is just `Adam` with a learning rate of $0.001$. 

```python
criterion = LabelSmoothingCrossEntropy()
criterion = criterion.to(device)
# note how the optimizer only optimizes model.head's parameters/
optimizer = optim.Adam(model.head.parameters(), lr=0.001)
```

The following code adds a learning rate scheduler, which tends to improve accuracy slightly. 

```python
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)
```

## Training

Let's define the training function. The following function will take the model, loss, optimizer, and the scheduler and train it for the specified number of epochs. In each epoch, the function does the following:

1. Start in the train phase of the epoch
   1. Set model in training mode `model.train()`
   2. Zero out the gradients
   3. Pass in the inputs to the model: `outputs = model(inputs)`
   4. Calculate the loss: `loss = criterion(outputs, labels`
   5. Backpropagate the loss: `loss.backward()`
   6. Update model weights: `optimizer.step()`
   7. Save loss/accuracy data and print it
2. Move to validation phase
   1. Set model to evaluation model `model.eval()`
   2. Turn off autograd (for faster inference) 
   3. Pass in the inputs to the model
   4. Calculate the loss
   5. Save loss/accuracy data and print it
3. Step the learning rate scheduler
4. Print the epoch training information
5. If validation accuracy is better than best validation accuracy, save the model weights.
6. Return the model with best validation accuracy.

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)
        
        for phase in ['train', 'val']: # We do training and validation phase per epoch
            if phase == 'train':
                model.train() # model to training mode
            else:
                model.eval() # model to evaluate
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): # no autograd makes validation go faster
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # used for accuracy
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step() # step at end of epoch
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc =  running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # keep the best validation accuracy model
        print()
    time_elapsed = time.time() - since # slight error
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Val Acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model
```

This should take around 5 minutes on a GPU-enabled Kaggle Kernel. 

```python
model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler)
```

```
[OUTPUT]
Epoch 0/9
----------
100%|██████████| 39/39 [00:22<00:00,  1.74it/s]
train Loss: 2.6152 Acc: 0.4325
100%|██████████| 8/8 [00:01<00:00,  6.73it/s]
val Loss: 1.6087 Acc: 0.7800
Epoch 1/9
----------
100%|██████████| 39/39 [00:17<00:00,  2.22it/s]
train Loss: 1.5363 Acc: 0.7516
100%|██████████| 8/8 [00:01<00:00,  7.44it/s]
val Loss: 1.2950 Acc: 0.8520
Epoch 2/9
----------
train Loss: 1.3366 Acc: 0.8303
val Loss: 1.2058 Acc: 0.9000
Epoch 3/9
----------
train Loss: 1.2571 Acc: 0.8593
val Loss: 1.1459 Acc: 0.9120
Epoch 4/9
----------
train Loss: 1.1892 Acc: 0.8878
val Loss: 1.0834 Acc: 0.9200
Epoch 5/9
----------
train Loss: 1.1495 Acc: 0.8989
val Loss: 1.0771 Acc: 0.9280
Epoch 6/9
----------
train Loss: 1.1123 Acc: 0.9112
val Loss: 1.0462 Acc: 0.9360
Epoch 7/9
----------
train Loss: 1.0940 Acc: 0.9140
val Loss: 1.0422 Acc: 0.9280
Epoch 8/9
----------
train Loss: 1.0719 Acc: 0.9243
val Loss: 1.0276 Acc: 0.9400
Epoch 9/9
----------
train Loss: 1.0533 Acc: 0.9296
val Loss: 1.0186 Acc: 0.9240
Training complete in 3m 7s
Best Val Acc: 0.9400
```

> I removed the progress bars after the first 2 epochs to save some space, you will see them during every epoch.

## Testing

The following code evaluates the model and gives its accuracy for each class. 

```python
test_loss = 0.0
class_correct = list(0 for i in range(len(classes)))
class_total = list(0 for i in range(len(classes)))
model_ft.eval()

for data, target in tqdm(test_loader):
    data, target = data.to(device), target.to(device)
    with torch.no_grad(): # turn off autograd for faster testing
        output = model_ft(data)
        loss = criterion(output, target)
    test_loss = loss.item() * data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    if len(target) == 32:
        for i in range(32):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

test_loss = test_loss / test_data_len
print('Test Loss: {:.4f}'.format(test_loss))
for i in range(len(classes)):
    if class_total[i] > 0:
        print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
            classes[i], 100*class_correct[i]/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])
        ))
    else:
        print("Test accuracy of %5s: NA" % (classes[i]))
print("Test Accuracy of %2d%% (%2d/%2d)" % (
            100*np.sum(class_correct)/np.sum(class_total), np.sum(class_correct), np.sum(class_total)
        ))
```

It should print the following, I get an accuracy of $93$%.

```
[OUTPUT]

100%|██████████| 8/8 [00:01<00:00,  6.62it/s]

Test Loss: 0.1136
Test Accuracy of adonis: 100% ( 4/ 4)
Test Accuracy of american snoot: 66% ( 2/ 3)
Test Accuracy of an 88: 100% ( 5/ 5)
Test Accuracy of banded peacock: 100% ( 5/ 5)
Test Accuracy of beckers white: 100% ( 4/ 4)
Test Accuracy of black hairstreak: 100% ( 5/ 5)
Test Accuracy of cabbage white: 100% ( 5/ 5)
Test Accuracy of chestnut: 100% ( 4/ 4)
Test Accuracy of clodius parnassian: 100% ( 4/ 4)
Test Accuracy of clouded sulphur: 100% ( 5/ 5)
Test Accuracy of copper tail: 100% ( 4/ 4)
Test Accuracy of crecent: 50% ( 2/ 4)
Test Accuracy of crimson patch: 100% ( 4/ 4)
Test Accuracy of eastern coma: 100% ( 5/ 5)
Test Accuracy of gold banded: 80% ( 4/ 5)
Test Accuracy of great eggfly: 80% ( 4/ 5)
Test Accuracy of grey hairstreak: 100% ( 5/ 5)
Test Accuracy of indra swallow: 100% ( 4/ 4)
Test Accuracy of julia: 100% ( 5/ 5)
Test Accuracy of large marble: 75% ( 3/ 4)
Test Accuracy of malachite: 80% ( 4/ 5)
Test Accuracy of mangrove skipper: 100% ( 5/ 5)
Test Accuracy of metalmark: 100% ( 5/ 5)
Test Accuracy of monarch: 100% ( 4/ 4)
Test Accuracy of morning cloak: 100% ( 5/ 5)
Test Accuracy of orange oakleaf: 100% ( 4/ 4)
Test Accuracy of orange tip: 100% ( 5/ 5)
Test Accuracy of orchard swallow: 100% ( 4/ 4)
Test Accuracy of painted lady: 100% ( 5/ 5)
Test Accuracy of paper kite: 100% ( 5/ 5)
Test Accuracy of peacock: 100% ( 5/ 5)
Test Accuracy of pine white: 100% ( 3/ 3)
Test Accuracy of pipevine swallow: 100% ( 5/ 5)
Test Accuracy of purple hairstreak: 50% ( 2/ 4)
Test Accuracy of question mark: 60% ( 3/ 5)
Test Accuracy of red admiral: 100% ( 5/ 5)
Test Accuracy of red spotted purple: 100% ( 5/ 5)
Test Accuracy of scarce swallow: 100% ( 5/ 5)
Test Accuracy of silver spot skipper: 100% ( 5/ 5)
Test Accuracy of sixspot burnet: 100% ( 5/ 5)
Test Accuracy of skipper: 100% ( 4/ 4)
Test Accuracy of sootywing: 50% ( 1/ 2)
Test Accuracy of southern dogface: 75% ( 3/ 4)
Test Accuracy of straited queen: 100% ( 4/ 4)
Test Accuracy of two barred flasher: 100% ( 5/ 5)
Test Accuracy of ulyses: 100% ( 5/ 5)
Test Accuracy of viceroy: 100% ( 3/ 3)
Test Accuracy of wood satyr: 100% ( 4/ 4)
Test Accuracy of yellow swallow tail: 80% ( 4/ 5)
Test Accuracy of zebra long wing: 100% ( 5/ 5)
Test Accuracy of 93% (210/224)
```

With more advanced training techniques and better models, one can get a higher accuracy

That's it for this article, see you in the next tutorial. 

Full Code can be found [here](https://www.kaggle.com/pdochannel/vision-transformers-in-pytorch-deit/data).
