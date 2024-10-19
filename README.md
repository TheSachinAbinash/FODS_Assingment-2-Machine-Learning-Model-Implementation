# Machine Learning Model Implementation

In this project, I have implemented the most popular Unsupervised Learning Technique called **K-Means Clustering** to classify an unlabelled dataset. I have used the *"superheroes_data"* from [Kaggle](https://www.kaggle.com/datasets/shreyasur965/super-heroes-dataset?resource=download) which contains a dataset of various fictional superheroes and their characteristics. My aim in this project was to classify superheroes of similar strengths into **10** distinct clusters, with **1** representing the *weakest superheroes* and *10* representing the *strongest superheroes*. 

I chose **K-Means Clustering** to perform this task since it is the most popular technique. Apart from this, I also used **Agglomerative Clustering** on the same dataset to cross-validate my results from **K-Means Clustering**. 

## Dataset 

Name : *superheroes_data* 

Link : [Kaggle](https://www.kaggle.com/datasets/shreyasur965/super-heroes-dataset?resource=download)

Description: *This dataset provides detailed information on 675 superheroes and villains from various comic universes. It includes a wide range of attributes such as power stats, biographical information, physical appearance, and affiliations. The data was collected using the SuperHero API, offering researchers and data enthusiasts a rich resource for analysis, machine learning projects, and comic book character studies.*

## Preprocessing

The dataset contained 731 entries and 26 features. First, I trimmed the to include only 7 features, namely *name*, *intelligence*, *strength*, *speed*, *durability*, *power*, *combat*, since these are only required features to perform clustering. Then, I used the **.is.null().sum()** to identify the number of null entries in each feature. After identification, I then used the **.dropna()** to drop the rows having these null entries. After that, I used the **df.duplicated()** to check for duplicate entries. After which I used the **.apply(lambda x : x.astype(str).str.lower()).drop_duplicates(subset=['name'], keep='first')** to drop all duplicate entries while only keeping the first instance. 

Having performed the above sequence of actions, the dataset is more or less used for the clustering technique. 

## Model Selection : K-Means Clustering and Agglomerative Clustering

**K-Means clustering** *is the most popular unsupervised learning algorithm. It is used when we have unlabelled data which is data without defined categories or groups. The algorithm follows an easy or simple way to classify a given data set through a certain number of clusters, fixed apriori. K-Means algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity.*

**Agglomerative clustering** *is the most frequent class of hierarchical clustering, used to put items in clusters based on similarities. In this approach, each item is first treated as a singleton cluster. Following this, pairs of clusters are merged one by one until all the clusters have been combined into a single large cluster holding all items. The output is a dendrogram, a tree-based presentation of the items.*

## Parameter Selection : Choosing the value of K. 

Both the K-Means Clustering and Agglomerative Clustering depends upon finding the number of clusters and data labels for a pre-defined value of K. To find the number of clusters in the data, we need to run the clustering algorithms for different values of K and compare the results. So, the performance of the model depends upon the value of K. We should choose the optimal value of K that gives us best performance. There are different techniques available to find the optimal value of K. The most common technique is the elbow method which is described below.

### The Elbow Method 

The elbow method is used to determine the optimal number of clusters in k-means clustering. The elbow method plots the value of the cost function produced by different values of k. If k increases, the average distortion will decrease. Then each cluster will have fewer constituent instances, and the instances will be closer to their respective centroids. However, the improvements in average distortion will decline as k increases. The value of k at which improvement in distortion declines the most is called the elbow, at which we should stop dividing the data into further clusters. 

## Model Evaluation 

Both the models in this projected have been evaluated using the **Davies-Bouldin Index**.

The Davies-Bouldin Index is a well-known metric to perform the same – it enables one to assess the clustering quality by comparing the average similarity between the pairwise most similar clusters. It is a validation metric that is used to evaluate clustering models. It is calculated as the average similarity measure of each cluster with the cluster most similar to it. In this context, similarity is defined as the ratio between inter-cluster and intra-cluster distances. As such, this index ranks well-separated clusters with less dispersion as having a better score.

## Conclusion

Having made use of K-Means Clustering, we have successfully classified the features into appropriate *Power Ranks* based on some intrinsic characteristics present in the dataset as identified by the model. We have also evaluated the proficiency of the model and have found it to be passable. Using the Agglomerative Clustering model to cross-validate our result, we find that our model is properly tuned and working. 
