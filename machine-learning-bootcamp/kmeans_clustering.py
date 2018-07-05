import matplotlib.pyplot as plt
from sklearn import cluster, datasets

iris = datasets.load_iris()
data = iris['data']

model = cluster.KMeans(n_clusters=3)
model.fit(data)

labels = model.labels_

label_data = data[labels == 0]
plt.scatter(label_data[:, 2], label_data[:, 3], c='black', alpha=0.3, s=100, marker='o')

label_data = data[labels == 1]
plt.scatter(label_data[:, 2], label_data[:, 3], c='black', alpha=0.3, s=100, marker='x')

label_data = data[labels == 2]
plt.scatter(label_data[:, 2], label_data[:, 3], c='black', alpha=0.3, s=100, marker='*')

plt.xlabel(iris['feature_names'][2])
plt.xlabel(iris['feature_names'][3])

plt.show()
