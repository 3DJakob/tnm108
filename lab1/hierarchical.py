import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage 

X = np.array([[5,3], [10,15], [15,12], [24,10], [30,30], [85,70], [71,80], [60,78], [70,55], [80,91] ])
#http://dellacqua.se/education/courses/tnm108/material/labs/Lab%201/shopping_data.csv

from sklearn.cluster import AgglomerativeClustering
customer_data = pd.read_csv('shopping_data.csv')
print(customer_data.shape)
print(customer_data.head())

data = customer_data.iloc[:, 3:5].values

linked = linkage(data, 'single')
labelList = range(1, 206)
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top',  distance_sort='descending', show_leaf_counts=True)
plt.show()


cluster = AgglomerativeClustering(n_clusters=7, affinity='euclidean', linkage='ward') 
cluster.fit_predict(data)

plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow') 
plt.show()