import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def cluster(X, with_PCA, method, name_method):
    print('K-means ' + with_PCA)

    scores = []
    n_clusters = [x for x in range(10, 301, 10)]
    
    for n_cluster in n_clusters:
        print(f'Applying for {n_cluster} clusters...')

        start_time = time.time()

        kmeans = method(n_clusters=n_cluster).fit(X)

        # each cluster has a list of lines of health.txt database
        dic = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

        health = pd.read_csv('health-dataset/health.txt', delimiter='|')
        headline_text = health['headline_text']

        f = open('DictionaryOfClusters' + name_method + with_PCA + '.txt', 'w+')
        for key in dic:
            f.write('Cluster ' + str(key) + '\n')
            f.write('\n'.join([str(x) + ': ' + headline_text[x] for x in dic[key]]) + '\n')
            f.write('----------------------------------------------------------------------------------------'  + '\n')

        # compute silhouette score
        print('Silhouette score:', silhouette_score(X, kmeans.labels_))
        
        # compute inertia
        print('Inertia:', kmeans.inertia_)
        scores.append(kmeans.inertia_)

        elapsed_time = time.time() - start_time
        print('Elapsed time: %1f s'%(elapsed_time))
        print()

    plot_scores(n_clusters, scores, 'GraphKMeans' + with_PCA + '.png')


def plot_scores(X, Y,  graph_name):
    fig, ax = plt.subplots()
    ax.plot(X, Y, label='Inertia')
    plt.ylabel('Score')
    plt.xlabel('Number of clusters')
    plt.title('Kmeans inertia_')
    ax.legend()
    plt.savefig(graph_name, bbox_inches='tight')
    plt.gcf().clear()


def main():
    bags = pd.read_csv('health-dataset/bags.csv', header=None)

    # clustering without PCA
    cluster(bags, '', KMeans, 'Kmeans')

    # clustering with PCA
    pca = PCA(.95)
    reducted_bags = pca.fit_transform(bags)
    cluster(reducted_bags, 'PCA', KMeans, 'Kmeans')


if __name__ == '__main__':
    main()