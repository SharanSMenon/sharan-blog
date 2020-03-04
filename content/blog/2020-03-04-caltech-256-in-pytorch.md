---
title: Caltech 256 in PyTorch
date: 2020-03-04T15:20:28.687Z
description: 'Transfer learning to classify the Caltech 256 dataset, in PyTorch (Densenet)'
---
Hello, we will classify the Caltech 256 dataset in Python using PyTorch, a deep learning framework

There are other alternatives like Tensorflow or Keras but we will use PyTorch in this article.  Let's get started!

## Getting the data

You can find the data here at this [Kaggle link](<>) or you could visit the [website](<>) itself. We will be using a Kaggle Kernel since the dataset is hosted on Kaggle and Kaggle kernels are an easy way to get started.

### Kaggle Kernel

Just head to the dataset link and create a new notebook. Make sure you turn **internet** on so that way we can download the model from the internet and you should turn the **GPU** on. 

> ### Why use GPU?
>
> You might be wondering, why use a GPU. GPU's speed up training by a lot. Models will train faster if placed on a GPU since GPU's are better suited for machine learning than CPU's. Kaggle Kernels and Google Colaboratory provide GPU's for free but availability might be limited.

If you run the first cell of your notebook (the first cell in newly created notebooks imports `numpy` and `pandas` and prints all the files in the input directory), you should see all the files in the input directory. You should also see the directory name of where they are stored. 

You might need to unzip the files before moving on so here is a snippet that will help you do that:

```python
!unzip  <path to zip file>
# The ! is important because it tells the 
# notebook you are running a shell command.
# Make sure to add the "!" at the beginning of
# the line if you are in a notebook.
```

Once you have unzipped the data and gotten it ready, we can now build a classifier.

If you don't want to use a Kaggle Kernel, other options include Google Colab or you can get a VM on something like GCP or AWS.

## Importing Modules

We need to import all our modules before we can get started. If you are in a Kaggle Kernel, you will notice that `numpy` and `pandas` are already imported for you. You won't be needing `pandas` for this project but `numpy` will be useful.

Let us now import the other modules that we need. 

We are importing modules like NumPy and matplotlib here. We also set the figure format to `retina` so that our images will be high res.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

Now we will import PyTorch and torchvision

```python
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
```

> Make sure that you have PyTorch and torchvision installed. Kaggle Kernels already have it preinstalled for you.

We also need to import PyTorch Ignite because we will be using it for training.

```python
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Accuracy, Loss, RunningAverage
```

> Make sure that you also have PyTorch Ignite installed. You will most likely not have it installed so go ahead and install it before continuing. Kaggle Kernels also have to Ignite preinstalled.

All right! All our modules are imported and we are ready to load the dataset

## Loading the Dataset

Let's load the dataset so that we can train the model. Here is what we are going to do:

1. Load the dataset
2. Resize the images to 224 x 224
3. Apply some other transforms
4. Split into training, testing, and validation.
5. Make batches of 64 images each for each set and convert the training, validation, and testing into iterators that return a batch.
6. Get the class names
7. Return the class names and the iterators

Let's start.

I have a utility function that will do all of the steps described above for us. 

```python
def get_data_loaders(data_dir, batch_size):
        transform = transforms.Compose([
		transforms.Resize(255),
		transforms.CenterCrop(224),
		transforms.ToTensor()
	])
	all_data = datasets.ImageFolder(data_dir,
							transform=transform)
	train_data_len = int(len(all_data)*0.75)
	valid_data_len = int((len(all_data) - train_data_len)/2)
	test_data_len = int(len(all_data) - train_data_len - valid_data_len)
	train_data, val_data, test_data = random_split(all_data, 
	[train_data_len, valid_data_len, test_data_len])
	train_loader = DataLoader(train_data, 
							  batch_size=batch_size, 
							  shuffle=True)
	val_loader = DataLoader(val_data, 
							batch_size=batch_size, 
							shuffle=True)
	test_loader = DataLoader(test_data, 
							batch_size=batch_size, 
							shuffle=True)
	return  ((train_loader, 
	val_loader, test_loader), 
			 all_data.classes)
```

We can use the method and load in the data

```python
(train_loader,val_loader,test_loader),classes = get_data_loaders(
	"/content/256_ObjectCategories", 
	 64)
# For Kaggle Kernels, the path is the following
# /kaggle/input/ Finish the path
```

Now we have our data loaded, we can build the classifier

## Building the Classifier

Like I said earlier, we will use transfer learning to classify the images and we will be loading in Densenet 121 as our model. You can use other models like ResNet, Inception, VGG, etc. if you want.

Here is the link to the paper that introduced Densenet to the world:

<https://arxiv.org/pdf/1608.06993.pdf>

One advantage to Densenet is that it has fewer parameters than models like Resnet and VGG, which make training faster and also the saved model has a smaller size. VGG models can be around 500 MB, while Resnet is 100mb and the model that we use, Densenet 121, is only 30 MB.

Densenet also achieves high accuracy with accuracies similar to VGG and ResNet.

Here is what we will do in this section

1. Load the model (will be retrained)
2. Freeze the parameters to make training quicker.
3. Change the number of output neurons in the last layer

Let's load in our model first:

```python
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = models.Densenet121(pretrained=True)
```

It will take a second or two to download. The model is around 30 MB. We also check if our GPU is available and if it is, we set the device to `CUDA`, otherwise, we use `CPU`. 

If you print the model by calling `print(model)`, you will see many layers and the last layer is called `classifier`. We need to modify that. This model was trained on imagenet so it returns 1000 classes. We only have 256 classes so we need to change the last layer to have 256 `out_features` instead of 1000. Let's do that now.

```python
for param in model.parameters():
	param.requires_grad = False  
# We have a pretrained model so we don't need to train 
# the entire model. 
# Just a small part of it needs to be trained.                    
```

We froze the mode above. We will now change the last layer below

```python
n_inputs = model.classifier.in_features
last_layer = nn.Linear(n_inputs,  len(classes))
model.classifier = last_layer
if torch.cuda.is_available():
	model.cuda()
print(model.classifier.out_features) # Should return 257
```

All right. We have changed the model to our needs and now we can train it. So go on to the next section to train the model. 

## Training the Model

Now that we have built the model, we can train it. This is were PyTorch Ignite comes in. We also need our GPU here. Here is what we will do in this section:

1. We will define the optimizer and the loss function
2. We will create a trainer and an evaluator.
3. We will write some functions to log training and validation losses and accuracies
4. We will train the model for around 5 epochs

### Defining the Loss Function and the Optimizer

We will use `CrossEntropyLoss` as our loss function and `Adam` with its default learning rate of `0.001` as our optimizer.

```python
criterion = nn.CrossEntropyLoss() # Loss function
optimizer = optim.Adam(model.classifier.parameters())
# Optimizer
```

### Training and validation history

We can collect our training and validation metrics so we can see how the model improved over time.  Here we define 2 dictionaries with 2 lists each that will hold the accuracy and loss for training and validation.

```python
training_history = {'accuracy':[],'loss':[]}
validation_history = {'accuracy':[],'loss':[]}
```

### Creating the trainer and evaluator

We are now creating our trainer and evaluator. The trainer will train the model while the evaluator evaluates the model and gives us some data like loss and accuracy.

```python
trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
evaluator = create_supervised_evaluator(model,
	device = device,
	metrics = {
		'accuracy': Accuracy(),
		'loss': Loss(criterion)
	}
)
```

### Creating logging functions

We will create functions to log the training and validation accuracy and epoch information.

```python
@trainer.on(Events.ITERATION_COMPLETED)
def log_a_dot(engine):
	print(".",end="")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
	evaluator.run(train_loader)
	metrics = evaluator.state.metrics
	accuracy = metrics['accuracy']*100
	loss = metrics['loss']
	training_history['accuracy'].append(accuracy)
	training_history['loss'].append(loss)
	print()
	print("Training Results - Epoch: {} Avg accuracy: {:.2f} Avg loss: {:.2f}"
	.format(trainer.state.epoch, accuracy, loss))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
	evaluator.run(val_loader)
	metrics = evaluator.state.metrics
	accuracy = metrics['accuracy']*100
	loss = metrics['loss']
	validation_history['accuracy'].append(accuracy)
	validation_history['loss'].append(loss)
	print("Validation Results - Epoch: {} Avg accuracy: {:.2f} Avg loss: {:.2f}"
	.format(trainer.state.epoch, accuracy, loss))
```

All right. Now we can finally train the model

### Training the model

All we need is 1 line to train the model.

```python
trainer.run(train_loader, max_epochs=5)
```

That's it. Just sit back and relax for a few minutes and you should have a fully trained model.

## Testing the model

Now that we have trained our model, we can test it. Testing the model is not too difficult. Just run the following line to evaluate the model:

```python
evaluator.run(test_loader)
metrics = evaluator.state.metrics
```

You can get accuracy off the metrics with the following line:

```python
accuracy = metrics['accuracy']*100
# multiplied by 100 for a percent between 1 
# and 00
print(accuracy) # Something like 91
```

If you trained the model right, your score should be above 80.

If you want to save the model, you can use a solution like TorchScript which also allows for the model to be loaded in other frontends, like C++.

## Conclusion

Well, we have created and trained a model that can classify the `caltech-256` dataset in PyTorch. If you want, here are a few bonus ideas for you to try.

### Things to try

Here are some ideas to try

* Use a different model, like Resnet or VGG
* Use a different optimizer
* Try the model out on custom images.
* Deploy the model
* Create an app that takes an image and classifies it
* Train on a different dataset, like Caltech birds or food 101

That's it for this post. I hope you enjoyed it and I will see you next time. Bye!
