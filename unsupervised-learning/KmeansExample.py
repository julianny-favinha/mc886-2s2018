import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1.5, 3], [2, 4], [4, 2], [3, 4], [4, 0], [5, 6], [7, 7]])
print(X[:,0])
print(X[:,1])
plt.scatter(X[:,0], X[:,1])

# compute kmeans clustering
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

# labels of each point
print("kmeans labels")
print(kmeans.labels_)

# predict the closest cluster each sample in X belongs to
print("kmeans predict")
print(kmeans.predict([[0, 0], [4, 4]]))

# coordinates of cluster centers
print("kmeans cluster centers")
print(kmeans.cluster_centers_)

dic = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
print(dic)

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color="red")
plt.show()