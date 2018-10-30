import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

def clustering(X, with_PCA):
    def apply(method, X):
        scores = []
        n_clusters = [x for x in range(5, 11, 5)]
        
        for n_cluster in n_clusters:
            print(f'Applying for {n_cluster} clusters...')
            kmeans = method(n_clusters=n_cluster).fit(X)

            # labels of each point
            # print('Labels:')
            # print(kmeans.labels_)

            # predict the closest cluster each sample in X belongs to
            # print("kmeans predict")
            # print(kmeans.predict([[0, 0], [4, 4]]))

            # coordinates of cluster centers
            # print('Cluster centers:')
            # print(kmeans.cluster_centers_)

            # each cluster has a list of lines of health.txt datase
            dic = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
            # print(dic)

            # compute silhouette score
            score = silhouette_score(X, kmeans.labels_)
            print("Silhouette score:", score)
            scores.append(score)

        return scores, n_clusters

    print('K-means ' + with_PCA)
    scores, n_clusters = apply(KMeans, X)
    plot(n_clusters, scores, 'GraphKMeans' + with_PCA + '.png')

    print('SpectralClustering' + with_PCA)
    scores, n_clusters = apply(SpectralClustering, X)
    plot(n_clusters, scores, 'GraphSpectralClustering' + with_PCA + '.png')

    print('AgglomerativeClustering' + with_PCA)
    scores, n_clusters = apply(AgglomerativeClustering, X)
    plot(n_clusters, scores, 'GraphAgglomerativeClustering' + with_PCA + '.png')


def plot(X, Y, graph_name):
    plt.plot(X, Y)
    plt.ylabel('Silhouette score')
    plt.xlabel('Number of clusters')
    plt.title('Elbow method')
    plt.savefig(graph_name, bbox_inches='tight')
    plt.gcf().clear()

def main():
    bags = pd.read_csv('health-dataset/bags.csv')
    # health = pd.read_csv('health-dataset/health.txt', delimiter='|')

    # clustering without PCA
    clustering(bags, "")

    # clustering with PCA
    pca = PCA(.95)
    reducted_bags = pca.fit_transform(bags)
    clustering(reducted_bags, "PCA")

if __name__ == "__main__":
    main()