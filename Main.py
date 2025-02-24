#import library
import pandas as pd

#read csv file
data=pd.read_csv('https://confrecordings.ams3.digitaloceanspaces.com/Market_Segmentation.csv')

#perform EDA
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.info())

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#write your code here

#make empty list
clusters = []

#plot elbow curve for 10 clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(data)
    clusters.append(kmeans.inertia_)
plt.plot(range(1,11),clusters)
plt.title('Elbow Method')
plt.xlabel("Num Clusters")
plt.ylabel('Inertia')
plt.show()

#Load KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0)
#Train model
kmeans.fit(data)

#Find centroids
centroids = kmeans.cluster_centers_
cluster_new=data.copy()

#predict
cluster_new['new_cluster_pred'] = kmeans.fit_predict(data)

#plot new clusters and centroids and show the plot
plt.scatter(cluster_new['Satisfaction'], cluster_new['Loyalty'], c=cluster_new['new_cluster_pred'], cmap='rainbow')
plt.plot(centroids[:,0], centroids[:,1],color='black')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()
