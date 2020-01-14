---
title: NLP with Pytorch
date: 2019-12-17T01:33:28.213Z
description: Classify a news dataset in python
---
We will do NLP using pytorch. This is text classification which will classify news articles. There are 4 types of articles that it can classify

1. Sports
2. Science/Tech
3. Business
4. World

## 1. Importing Modules

We are importing the modules like pytorch and torch text. We are also importing the text classification dataset

```python
import torch
import torchtext
```

```python
from torchtext.datasets import text_classification
```

## 2. Getting the dataset

We actually load the dataset here

```python
NGRAMS = 2
import os
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU or CPU
```

```
ag_news_csv.tar.gz: 11.8MB [00:03, 3.13MB/s]
120000lines [00:09, 12467.27lines/s]
120000lines [00:21, 5659.35lines/s]
7600lines [00:01, 4520.54lines/s]
```

## 3. Building the model

### 3.1 Importing the NN Module

```python
import torch.nn as nn
import torch.nn.functional as F
```

### 3.2 Making the Neural Network Class

```python
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
```

## 4. Initializing the model

Initializes the model and some parameters

```python
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
```

```python
model # Displaying the model
```

```
TextSentiment(
  (embedding): EmbeddingBag(1308844, 32, mode=mean)
  (fc): Linear(in_features=32, out_features=4, bias=True)
)
```

## 5. Training functions

```python
def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label
```

```python
from torch.utils.data import DataLoader
```

```python
def train_func(sub_train_):
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()
    scheduler.step()
    return train_loss / len(sub_train_), train_acc / len(sub_train_)
def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)
```

## 6. Training the model

```python
import time
from torch.utils.data.dataset import random_split
```

### 6.1 Initializing some variables and functions

We initialize the number of epochs, and the optimizer and some other things in the next 3 cells

```python
N_EPOCHS = 5
min_valid_loss = float('inf')
```

```python
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
```

```python
train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])
```

### 6.2 The Training loop

In the following cell, this is where all the training happens

```python
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
    
```

```
Epoch: 1  | time in 0 minutes, 37 seconds
	Loss: 0.0263(train)	|	Acc: 84.6%(train)
	Loss: 0.0001(valid)	|	Acc: 90.8%(valid)
Epoch: 2  | time in 0 minutes, 38 seconds
	Loss: 0.0119(train)	|	Acc: 93.6%(train)
	Loss: 0.0001(valid)	|	Acc: 91.3%(valid)
Epoch: 3  | time in 0 minutes, 40 seconds
	Loss: 0.0070(train)	|	Acc: 96.3%(train)
	Loss: 0.0001(valid)	|	Acc: 91.4%(valid)
Epoch: 4  | time in 0 minutes, 34 seconds
	Loss: 0.0039(train)	|	Acc: 98.0%(train)
	Loss: 0.0001(valid)	|	Acc: 91.7%(valid)
Epoch: 5  | time in 0 minutes, 35 seconds
	Loss: 0.0023(train)	|	Acc: 99.0%(train)
	Loss: 0.0001(valid)	|	Acc: 91.7%(valid)
```

## 7. Evaluating the model

We just run the `test` function on the dataset. and print it out

```python
print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
```

```
Checking the results of test dataset...
	Loss: 0.0002(test)	|	Acc: 91.2%(test)
```

## 8. Trying it out

Here we can try out the model

### 8.1 Importing and initializing variables

We import some more libraries like `re` and we make the labels. There are 4 labels as mentioned at the top of the notebook

```python
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
```

```python
ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}
```

### 8.2 Building the prediction function

```python
def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1
```

### 8.3 Predicting on some text

```python
ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])
```

```
This is a Sports news
```

## 9. Conclusion

We have built a classifier that classifies 4 types of news. It has a 98% train accuracy and a 91% test accuracy. That is a pretty good model.
