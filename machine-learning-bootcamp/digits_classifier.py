import numpy as np
from sklearn import datasets, tree, metrics

digits = datasets.load_digits()
# print("digits: {0}".format(digits))

# List that stores location of data of 3 and 8
flag_3_8 = (digits.target == 3) + (digits.target == 8)
# print("flag_3_8: {0}".format(flag_3_8))

images = digits.images[flag_3_8]
labels = digits.target[flag_3_8]

# 1-dimensioning
images = images.reshape(images.shape[0], -1)

n_samples = len(flag_3_8[flag_3_8])
# print("flag_3_8[flag_3_8]: {0}".format(flag_3_8[flag_3_8]))
train_size = int(n_samples * (3/5))
classifier = tree.DecisionTreeClassifier()
classifier.fit(images[:train_size], labels[:train_size])

expected_data = labels[train_size:]
predicted_data = classifier.predict(images[train_size:])

print("Accuracy: ", metrics.accuracy_score(expected_data, predicted_data))