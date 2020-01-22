---
title: Glass Classification in PyTorch
date: 2020-01-22T17:55:56.754Z
description: Classify Glass types in PyTorch (and a bonus)
---
Classifiying glass based on properties. Here is the [Kaggle](https://www.kaggle.com/uciml/glass/) url to download the dataset

We will use PyTorch and build a neural network. I have also included a small bonus at the end.

## Importing Modules

We need numpy for working with the data (and other tasks like training) and we need pandas for reading the data

```python
import numpy as np
import pandas as pd
```

### Data viz modules

We import `matplotlib` and `seaborn` for data visualization.

We also use some cell magic so that we can see visualizations in the notebook and to increase the quality of the image

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

## Loading Data

```python
df = pd.read_csv("../../data/glass.csv") # Replace the path the path to your file
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RI</th>
      <th>Na</th>
      <th>Mg</th>
      <th>Al</th>
      <th>Si</th>
      <th>K</th>
      <th>Ca</th>
      <th>Ba</th>
      <th>Fe</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.52101</td>
      <td>13.64</td>
      <td>4.49</td>
      <td>1.10</td>
      <td>71.78</td>
      <td>0.06</td>
      <td>8.75</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.51761</td>
      <td>13.89</td>
      <td>3.60</td>
      <td>1.36</td>
      <td>72.73</td>
      <td>0.48</td>
      <td>7.83</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.51618</td>
      <td>13.53</td>
      <td>3.55</td>
      <td>1.54</td>
      <td>72.99</td>
      <td>0.39</td>
      <td>7.78</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.51766</td>
      <td>13.21</td>
      <td>3.69</td>
      <td>1.29</td>
      <td>72.61</td>
      <td>0.57</td>
      <td>8.22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.51742</td>
      <td>13.27</td>
      <td>3.62</td>
      <td>1.24</td>
      <td>73.08</td>
      <td>0.55</td>
      <td>8.07</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

Our feature columns are `[Ri, Na, Mg, Al, Si, K, Ca, Ba, Fe]` and our target column is `Type`

1. `Ri` stands for refractive index
2. `Na` stands for the amount of sodium
3. `Mg` stands for the amount of magnesium
4. `Al` stands for the amount of aluminum
5. `Si` stands for the amount of silicon
6. `K` stands for the amount of Potassium
7. `Ca` stands for the amount of Calcium
8. `Ba` stands for the amount of Barium
9. `Fe` stands for the amount of Iron

There are 7 types: `building_windows_float_processed`, `building_windows_non_float_processed`, `vehicle_windows_float_processed`, `vehicle_windows_non_float_processed` (None here), `containers`, `tableware`, `headlamps`.

We will try to classify the 7 types using a neural network, except for the type that is not there. So we basically have 6 types, and this will be important

## Exploratory Data Analysis

Let's explore the data a bit by making some graphs.

```python
df.hist(column='Type', bins=7, figsize=(14, 6)) # Make the figure size smaller if needed
# 7 Bins for the 7 classes. We can see that there is no rows with a type of "4"
```

```
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1a1e9e1890>]],
      dtype=object)
```

![png](/img/2020-01-22-at-12.54.01-pmscreenrecording_.png)

> You can skip the following cell because it will take some time to run unless you want to see the output.

```python
sns.pairplot(df, hue='Type')
```

```
<seaborn.axisgrid.PairGrid at 0x1a1f136b50>
```

![png](/img/2020-01-22-at-12.54.01-pmscreenrecording_-2.png)

```python
fig = plt.figure(figsize=(14, 6))
sns.distplot(df['Type'], bins=7) # We can see the distribution of the types.
```

```
<matplotlib.axes._subplots.AxesSubplot at 0x1a22f62810>
```

![png](/img/2020-01-22-at-12.54.01-pmscreenrecording_-3.png)

There are a lot of rows that fall as either 1 or 2 while there isn't as much data for types 5 - 7. 

> Type 8 and 0 don't exist. It is just there

We can expect that our model is more likely to predict something as 1 or 2 than any other category

## Building the model

Let's build the model now, but first we will split the data into testing and training. We will use the training data to train the model and testing data to test how well the model does on unseen data

### Splitting to testing and training

We will split it into training and testing in the following cells

```python
from sklearn.model_selection import train_test_split
```

```python
X = df.drop('Type', axis=1)
y = df['Type'].apply(lambda x: x-1)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

```python
print(X_train.shape)
print(y_train.shape)
```

```
(143, 9)
(143,)
```

### Importing PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

### Converting the data to PyTorch tensors

We can't feed numpy arrays and lists into the neural network. We have to feed PyTorch Tensors. So that's why we are converting everything into a tensor. The targets are `long` tensors because they are all integers while the inputs have `float` values

```python
X_train_tensor = torch.tensor(X_train.values).float()
X_test_tensor = torch.tensor(X_test.values).float()
y_train_tensor = torch.tensor(y_train.values).long()
y_test_tensor = torch.tensor(y_test.values).long()
```

### Building the Model

We define the model architecture here. I will explain each line in detail.

```python
class GlassClassifier(nn.Module):
    def __init__(self):
        super(GlassClassifier, self).__init__()
        self.fc1 = nn.Linear(9, 64) # 9 Input features
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 7)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
```

The model has 3 fully connected layers, `fc1`, `fc2`, `fc3`, `fc4`. 

`fc1` receives the input while `fc2` and `fc3` are hidden layers. `fc4` is the output layer and this returns the output. These layers are defined in the `__init__` function.

#### The `forward` function

The forward function is where the forward pass takes place. When we give the neural network data, it goes through the `forward` function.

```python
x = F.relu(self.fc1(x)) # Don't add this code - Just for explanation
```

We are sending the data through the fully connnected layer and then using a **relu** activation function on it.

```python
model = GlassClassifier()
print(model) # You can add a option 

if torch.cuda.is_available():
    model.cuda()
```

```
GlassClassifier(
  (fc1): Linear(in_features=9, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=128, bias=True)
  (fc4): Linear(in_features=128, out_features=7, bias=True)
)
```

## Training the Model

### Loss and Optimizer

I have chosen `CrossEntropyLoss` as the loss function and `Adam` as the optimizer. I also found that Adam works better than SGD for this task

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
```

### Training

We train for 1000 epochs. You can choose any number of epochs, I recommend above 400 epochs. Below is the training loop. This is a pretty standard PyTorch training loop. We should see our loss decrease to around `0.26` near the end of the training loop. We are also saving the model with the lowest loss because the loss sometimes spikes and we don't want our model to be affected by the spikes.

```python
n_epochs = 1000
losses = [] # Collecting all the losses
lowest_loss = np.Inf # For saving the model.
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if loss.item() < lowest_loss: 
        # If the loss is lower than the lowest loss,
        # than the model gets saved
        print("Loss decreased: {} -> {}".format(lowest_loss, loss.item()))
        torch.save(model.state_dict(), "glass-classification.pt")
        lowest_loss = loss.item()
    print('Epoch {}, Loss {}'.format(epoch, loss.item()))
```

```
Loss decreased: inf -> 2.0952165126800537
Epoch 1, Loss 2.0952165126800537
Epoch 2, Loss 2.0992839336395264
Epoch 3, Loss 2.8283398151397705
Loss decreased: 2.0952165126800537 -> 2.0399701595306396
......
Epoch 998, Loss 0.2502361536026001
Epoch 999, Loss 0.3026759624481201
Epoch 1000, Loss 0.2624070346355438
```

```python
lowest_loss # Printing the lowest loss
```

```
0.24013982713222504
```

```python
model.load_state_dict(torch.load("glass-classification.pt"))
# We can load the model with the lowest loss from this line.
```

```
<All keys matched successfully>
```

### Plotting the losses.

We plot the losses here and we can see a few spikes here and there

```python
plt.plot(losses)
```

```
[<matplotlib.lines.Line2D at 0x1a25fa2d10>]
```

![png](/img/output_36_1.png)

## Evaulating the model

We will now evaulate our neural network and I get a score of 70%. We use scikit-learn's metrics including the `accuracy_score` and the `classification_report`.

```python
preds_test = model(X_test_tensor)
_, preds = torch.max(preds_test, 1)
```

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

```python
accuracy_score(y_test, preds.detach().numpy())
```

```
0.676056338028169
```

```python
print(classification_report(y_test, preds.detach().numpy()))
```

```
              precision    recall  f1-score   support

           0       0.72      0.75      0.73        24
           1       0.79      0.63      0.70        30
           2       0.00      0.00      0.00         4
           4       0.25      1.00      0.40         2
           5       0.50      1.00      0.67         2
           6       1.00      0.78      0.88         9

    accuracy                           0.68        71
   macro avg       0.54      0.69      0.56        71
weighted avg       0.73      0.68      0.69        71
```

We also have a confusion matrix here displayed as a heatmap

```python
fig, ax = plt.subplots(figsize=(10,6))  
ax= plt.subplot()
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels') 
ax.set_title('Confusion Matrix') 
sns.heatmap(confusion_matrix(y_test, preds.detach().numpy()), 
            annot=True, ax=ax, fmt="d")
```

```
<matplotlib.axes._subplots.AxesSubplot at 0x1a274e4450>
```

![png](/img/2020-01-22-at-12.54.01-pmscreenrecording_-5.png)

## Bonus

Here is the bonus section. We will train a scikit-learn classifier on this same dataset. We shall use a random forest classifier here

```python
from sklearn.ensemble import RandomForestClassifier # Importing the random forest classifier
```

### Building the model

The following cell builds the model

```python
rfc = RandomForestClassifier(n_estimators=150) # Very easy to make a RFC
```

### Training the model

This is how you train a RFC

```python
rfc.fit(X_train, y_train) # Also
```

```
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=150,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
```

### Evaulating the model

We are using the rfc to predict unseen data

```python
preds_rfc = rfc.predict(X_test)
```

We have already imported these earlier. You can also see that training and evaulating a RFC is really easy, way easier than the neural network

```python
accuracy_score(y_test, preds_rfc)
```

```
0.704225352112676
```

```python
print(classification_report(y_test, preds_rfc)) # More detailed classification report
```

```
              precision    recall  f1-score   support

           0       0.79      0.92      0.85        24
           1       0.81      0.57      0.67        30
           2       0.20      0.25      0.22         4
           4       0.29      1.00      0.44         2
           5       0.50      1.00      0.67         2
           6       1.00      0.67      0.80         9

    accuracy                           0.70        71
   macro avg       0.60      0.73      0.61        71
weighted avg       0.77      0.70      0.71        71
```

## Conclusion

Our random forest classifier did a bit better than our neural network, but they both did well and thats it for this tutorial. I hope you understood how neural networks work and how you can build them in pytorch (and how you can build Random Forest Classifiers in scikit-learn). That's it for now and I will see you next time.
