import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

def medoids(X, kmeans, name_method, with_PCA, headline_text, dic):

    f = open('MedoidsOfClusters' + name_method + with_PCA + '.txt', 'w+')
    
    for cluster in dic:
        lines = []
        if(len(dic[cluster]) > 3):
            cluster_elems = []
            cluster_values = []
            for elem in dic[cluster]:
                cluster_elems.append(elem)
                cluster_values.append(X.values[elem])
            distances = euclidean_distances(cluster_values, [kmeans.cluster_centers_[cluster]], squared=True)

            mins = np.argsort(distances.flatten())[:3]
            lines = [cluster_elems[x] for x in mins]
           
        else:
            lines = dic[cluster]

        f.write('Medoids Cluster ' + str(cluster) + '\n')
        f.write('\n'.join([str(x) + ': ' + headline_text[x] for x in lines]) + '\n')
        f.write('----------------------------------------------------------------------------------------'  + '\n')
    f.close()   


def cluster(X, with_PCA, method, name_method):
    print('K-means ' + with_PCA)

    scores = []
    n_clusters = [x for x in range(300, 301, 10)]
    
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
        f.close()

        medoids(X, kmeans, name_method, with_PCA, headline_text, dic)

        # compute silhouette score
        print('Silhouette score:', silhouette_score(X, kmeans.labels_))
        
        # compute inertia
        print('Inertia:', kmeans.inertia_)
        scores.append(kmeans.inertia_)

        elapsed_time = time.time() - start_time
        print('Elapsed time: %1f s'%(elapsed_time))
        print()

    plot_scores(n_clusters, scores, with_PCA, 'GraphKMeans' + with_PCA + '.png')


def plot_scores(X, Y, with_PCA, graph_name):
    fig, ax = plt.subplots()
    ax.plot(X, Y, label='Inertia')
    plt.ylabel('Score')
    plt.xlabel('Number of clusters')
    plt.title('Kmeans ' + with_PCA + ' inertia_')
    ax.legend()
    plt.savefig(graph_name, bbox_inches='tight')
    plt.gcf().clear()


def main():
    bags = pd.read_csv('health-dataset/bags.csv', header=None)

    # clustering without PCA
    cluster(bags, '', KMeans, 'Kmeans')

    # clustering with PCA
    # pca = PCA(.95)
    # reducted_bags = pca.fit_transform(bags)
    # cluster(reducted_bags, 'PCA 0 95', KMeans, 'Kmeans')

    # pca = PCA(.65)
    # reducted_bags = pca.fit_transform(bags)
    # cluster(reducted_bags, 'PCA 0 65', KMeans, 'Kmeans')


if __name__ == '__main__':
    main()