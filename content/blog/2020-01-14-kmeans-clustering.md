---
title: KMeans Clustering
date: 2020-01-14T15:52:20.167Z
description: KMeans Clustering in Python using Scikit-learn
---
Happy New Year!  It is 2020 now. I am sorry for the long break I took from writing, I was on vacation. 

This tutorial will talk about KMeans clustering in scikit-learn and python. Let's do KMeans clustering with python.

In the below cells, I am importing libraries and setting up matplotlib

> Wherever it says `(out)`, that is output

```python
import seaborn as sns
```

```python
import matplotlib.pyplot as plt
```

```python
%matplotlib inline
plt.style.use("ggplot")
```

We will be creating our dataset so we need to import the `make_blobs` function. In the cell after that, we create the data for clustering

```python
from sklearn.datasets import make_blobs
```

```python
data = make_blobs(n_samples=200, n_features=2, centers=4,
                 cluster_std=1.8,
                 random_state=101)
```

```python
data[0].shape
```

```
(Out) (200, 2)
```

We plot the data and we can see the clusters

```python
plt.figure(figsize=(10, 6))
plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap="rainbow")
```

![graph of KMeans](/img/output_9_1.png)

Here is where we import the KMeans clustering algorithm. It is very easy to work with KMeans clustering in Python.

```python
from sklearn.cluster import KMeans
```

We initialize and train the KMeans clustering algorithm in the following cells

```python
kmeans = KMeans(n_clusters=4) # We know about the number of clusters here since we specified it earlier
```

```python
kmeans.fit(data[0]) #
```

```
(Out) KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
```

We can see what the algorithm thinks the cluster centers are and what the cluster labels are for each point.

```python
kmeans.cluster_centers_
```

```
(Out) array([[ 3.71749226,  7.01388735],
       [-9.46941837, -6.56081545],
       [-0.0123077 ,  2.13407664],
       [-4.13591321,  7.95389851]])
```

```python
kmeans.labels_
```

```
(Out) array([3, 0, 2, 0, 0, 1, 0, 2, 0, 2, 3, 2, 0, 0, 3, 2, 0, 2, 1, 3, 1, 2,
       2, 1, 3, 1, 1, 2, 0, 0, 3, 1, 0, 2, 2, 3, 1, 1, 1, 2, 1, 3, 3, 3,
       2, 0, 3, 2, 1, 2, 2, 3, 0, 2, 1, 3, 2, 2, 3, 0, 1, 0, 1, 3, 0, 2,
       1, 0, 0, 1, 0, 2, 1, 2, 1, 0, 0, 2, 3, 2, 2, 1, 0, 1, 2, 2, 2, 3,
       2, 1, 1, 1, 1, 2, 2, 1, 0, 3, 1, 0, 2, 1, 2, 2, 0, 2, 1, 0, 1, 1,
       0, 3, 3, 0, 1, 0, 3, 3, 0, 3, 2, 3, 2, 3, 2, 0, 3, 2, 1, 3, 3, 3,
       2, 1, 1, 3, 0, 3, 0, 2, 1, 0, 1, 3, 3, 0, 2, 1, 3, 3, 3, 3, 2, 0,
       2, 3, 0, 0, 0, 2, 0, 2, 2, 3, 1, 3, 2, 0, 3, 2, 0, 2, 3, 0, 2, 3,
       0, 0, 1, 0, 3, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 0, 0, 3, 1, 2, 0, 0,
       1, 2], dtype=int32)
```

Let's compare the KMeans result with the actual data. We will plot them both side by side and we can see that the clustering algorithm does pretty well.. It gets most of the data points right. It gets the blue cluster correct and it does well on the other 3 clusters except for those points in the center.

> Ignore the colors. They can be a bit confusing

```python
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

ax1.set_title('K Means')
ax1.scatter(data[0][:,0], data[0][:,1], c=kmeans.labels_, cmap="rainbow")
ax2.set_title('Actual Data')
ax2.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap="rainbow")
```

```
<matplotlib.collections.PathCollection at 0x119cecf60>
```

![KMeans clustering vs Actual data](/img/output_19_1.png)

So that's it. We have done KMeans clustering in Python using scikit-learn and you can see that it is really accurate. 

Well I hope you enjoyed this tutorial and I will see you next time. Bye!
