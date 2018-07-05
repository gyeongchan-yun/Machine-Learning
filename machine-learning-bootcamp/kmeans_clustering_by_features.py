import matplotlib.pyplot as plt
from sklearn import cluster, datasets

iris = datasets.load_iris()
data = iris['data']

model = cluster.KMeans(n_clusters=3)
model.fit(data)

labels = model.labels_

MARKERS = ['o', 'x', '*']

def scatter_by_feature(idx1, idx2):
    for label_idx in range(labels.max() + 1):
        clustered_data = data[labels == label_idx]
        plt.scatter(clustered_data[:, idx1], clustered_data[:, idx2],
                    c='black', alpha=0.3, s=100, marker=MARKERS[label_idx], label='label {}'.format(label_idx))

    plt.xlabel(iris['feature_names'][idx1], fontsize='xx-large')
    plt.ylabel(iris['feature_names'][idx2], fontsize='xx-large')

plt.figure(figsize=(16, 16))

plt.subplot(3, 2, 1)
scatter_by_feature(0, 1)

plt.subplot(3, 2, 2)
scatter_by_feature(0, 2)

plt.subplot(3, 2, 3)
scatter_by_feature(0, 3)

plt.subplot(3, 2, 4)
scatter_by_feature(1, 2)

plt.subplot(3, 2, 5)
scatter_by_feature(1, 3)

plt.subplot(3, 2, 6)
scatter_by_feature(2, 3)

plt.tight_layout()
plt.show()
