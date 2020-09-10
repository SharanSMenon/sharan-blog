---
title: FastAI Image Classifier
date: 2020-09-10T15:37:09.689Z
description: Build a image classifier using Fast.AI
---
In this tutorial, you will learn how to build an image classifier in under 10 lines of code. This classifier will learn to differentiate between cats and dogs.

## Setup

Make sure that you have PyTorch and fast.ai installed. If you want to install PyTorch, check out their [website](https://pytorch.org). To install fast.ai, just run the following command:

```shell
pip install fastai
```

There we go. Now we can get started with creating the classifier.

> If you are running this notebook in Google Colab, you will need to update fast.ai to version 2. Just run `!pip install -U fastai` in a cell and you should be set.


```python
from fastai.vision.all import *
# This is the only import we need
```

The following import above contains everything we need.

## Data

We need to get the data first. That is a really simple process.

```python
path = untar_data(URLs.PETS)/'images'
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=42, label_func=is_cat, item_tfms=Resize(224))
```

The lines above load the data, creates some labels, and procesees the data

## Model and Training

Now we will load and train the model. Here is the code to do that.

```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```

The first line loads the ResNet model and passes in the data. We also tell it that we want to see the error rate. The second line trains the model. This will take a minute or 2, depending on your machine. Once it is done, you can go to the next section

## Inference

If you want to try out the model on a new image, you can use the `.predict` method. Here is the code for that.

```python
img = PILImage.create('cat.jpg') # Replace
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```

First line loads the image. Second line passes the image into the model and gets the probability that its a cat or dog. 3rd and 4th lines just print the results.

## Full Code

Here is the full code for this program.

```python
from fastai.vision import *
path = untar_data(URLs.PETS)/'images'
def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=42, label_func=is_cat, item_tfms=Resize(224))
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
img = PILImage.create('cat.jpg') # Replace
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```

This program only takes 10 lines. But, we loaded a dataset, preprocessed it, loaded in a pretrained ResNet model, and trained it to a high accuracy rate.
