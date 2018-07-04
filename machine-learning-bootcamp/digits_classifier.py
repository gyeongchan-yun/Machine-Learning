import numpy as np
from sklearn import datasets, tree, metrics, ensemble, svm

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
        # == parameter
        #    - n_estimators: number of weak learners
        #    - criterion: decision tree algorithm
        # == random forest - bagging
        classifier = ensemble.RandomForestClassifier(n_estimators=20, max_depth=3, criterion="gini")
    elif index == 3:
        # == parameter
        #    - base_estimator: decide weak learners
        # == adaboost - boosting
        classifier = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=3), n_estimators=20)
    elif index == 4:
        # == parameter
        #    - C: error parameter. How much error is allowed
        #    - gamma: The higher value is, the more complicated curved surface is determined (곡면)
        classifier = svm.SVC(C=1.0, gamma=0.001)
    else:
        print("No classifier\n")
        sys.exit(0)

    implement_classifier(classifier)

if __name__ == "__main__":
    main()