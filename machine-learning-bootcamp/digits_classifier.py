import numpy as np
from sklearn import datasets, tree, metrics, ensemble

import sys


def implement_classifier(classifier):
    digits = datasets.load_digits()
    # print("digits: {0}".format(digits))

    # == List that stores location of data of 3 and 8
    flag_3_8 = (digits.target == 3) + (digits.target == 8)
    # print("flag_3_8: {0}".format(flag_3_8))

    images = digits.images[flag_3_8]
    labels = digits.target[flag_3_8]

    # == 1-dimensioning
    images = images.reshape(images.shape[0], -1)

    n_samples = len(flag_3_8[flag_3_8])
    # print("flag_3_8[flag_3_8]: {0}".format(flag_3_8[flag_3_8]))
    train_size = int(n_samples * (3/5))

    classifier.fit(images[:train_size], labels[:train_size])

    expected_data = labels[train_size:]
    predicted_data = classifier.predict(images[train_size:])

    # == Performance of classifier
    print("Accuracy: ", metrics.accuracy_score(expected_data, predicted_data))

    print("\nConfusion Matrix: \n", metrics.confusion_matrix(expected_data, predicted_data))

    print("\nPrecision: ", metrics.precision_score(expected_data, predicted_data, pos_label=3))

    print("\nRecall: ", metrics.recall_score(expected_data, predicted_data, pos_label=3))

    print("\nF-measure: ", metrics.f1_score(expected_data, predicted_data, pos_label=3))


def main():
    if len(sys.argv) < 2:
        print("Usage: python digits_classifier.py index")
        sys.exit(0)

    index = int(sys.argv[1])

    if index == 0:
        classifier = tree.DecisionTreeClassifier()
    elif index == 1:
        classifier = tree.DecisionTreeClassifier(max_depth=3)
    elif index == 2:
        # == n_estimators: number of weak learners, criterion: decision tree algorithm
        # == random forest - bagging algorithm 
        classifier = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, criterion="gini")
    else:
        print("No classifier\n")
        sys.exit(0)

    implement_classifier(classifier)

if __name__ == "__main__":
    main()