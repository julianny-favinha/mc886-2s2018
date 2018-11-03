import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def cluster(X, with_PCA, method, name_method):
    print('DBSCAN ' + with_PCA)

    scores = []
    epss = [0.05, 0.1, 0.719, 0.769, 0.788, 1.0]
    min_sampless = [2, 3, 4, 5]

    for eps in epss:
        for min_samples in min_sampless:
            print(f'Applying DBSCAN for eps = {eps} and min_samples = {min_samples}...')

            start_time = time.time()

            dbscan = method(eps=eps, min_samples=min_samples).fit(X)

            # number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
            print('Number of clusters:', n_clusters)

            # each cluster has a list of lines of health.txt database
            dic = {i: np.where(dbscan.labels_ == i)[0] for i in range(n_clusters)}

            health = pd.read_csv('health-dataset/health.txt', delimiter='|')
            headline_text = health['headline_text']

            f = open('DictionaryOfClusters' + name_method + with_PCA + '.txt', 'w+')
            for key in dic:
                f.write('Cluster ' + str(key) + '\n')
                f.write('\n'.join([str(x) + ': ' + headline_text[x] for x in dic[key]]) + '\n')
                f.write('----------------------------------------------------------------------------------------'  + '\n')

            # removing outliers (labelled in cluster -1)
            X_new = X.copy()
            X_new['labels_'] = dbscan.labels_
            X_new = X_new[(X_new.labels_ != -1)]
            print('Number of observations which is not outlier:', X_new.shape[0])

            labels = [x for x in dbscan.labels_ if x != -1]

            # compute silhouette score
            print('Silhouette score:', silhouette_score(X_new, labels))
            
            elapsed_time = time.time() - start_time
            print('Elapsed time: %1f s'%(elapsed_time))
            print()


def main():
    bags = pd.read_csv('health-dataset/bags.csv', header=None)

    # clustering without PCA
    cluster(bags, '', DBSCAN, 'DBSCAN')

    # clustering with PCA
    # pca = PCA(.95)
    # reducted_bags = pca.fit_transform(bags)
    # cluster(reducted_bags, 'PCA', DBSCAN, 'DBSCAN')


if __name__ == '__main__':
    main()