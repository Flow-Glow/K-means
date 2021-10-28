# K Means Algorithm

## What is K Means
  This algorithm is an iterative algorithm that partitions the dataset according to their features into K number of predefined non- overlapping distinct clusters or subgroups. It makes the data points of inter clusters as similar as possible and also tries to keep the clusters as far as possible. It allocates the data points to a cluster if the sum of the squared distance between the cluster’s centroid and the data points is at a minimum, where the cluster’s centroid is the arithmetic mean of the data points that are in the cluster. A less variation in the cluster results in similar or homogeneous data points within the cluster.
### Sources :
1. [Nvidia](https://www.nvidia.com/en-us/glossary/data-science/k-means/#:~:text=K%20Means%20is%20one%20of%20the%20simplest%20and,that%20haven%E2%80%99t%20been%20explicitly%20labeled%20within%20the%20data.)
2. [Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
- - - -
## How K Means works
1. Specify number of clusters K.
2. Initialize centroids by first shuffling the dataset and then randomly selecting K data points for the centroids without replacement.
3. Keep iterating until there is no change to the centroids. i.e assignment of data points to clusters isn’t changing.
4. Compute the euclidean distance
5. Assign each data point to the closest cluster (centroid).
6. Compute the centroids for the clusters by taking the average of the all data points that belong to each cluster.
### Flow Chart
![picture alt](https://i.gyazo.com/5271a8a2172de9246c69e038aac3cffa.png "Flow Chart")
- - - -
## K Means in action
### 2D:
![picture alt](https://i.gyazo.com/d44920e3c9eb6aeb596d19adc81bbf4c.png "K Means in 2D")
### 3D:
![picture alt](https://i.gyazo.com/8b21d8e5a01cceb0f8167a5227b3dd78.png "K Means in 3D")

