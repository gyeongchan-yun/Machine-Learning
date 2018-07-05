import matplotlib.pyplot as plt
import  numpy as np

import math
import sys

from sklearn import linear_model, svm, ensemble, neighbors


def create_data():
    x = np.random.rand(1000, 1)
    x = x * 20 - 10

    y = np.array([math.sin(a) for a in x])  # sin curve
    y += np.random.randn(1000)  # noise

    return x, y


def create_model(method):
    if method == 0:
        return linear_model.LinearRegression()
    elif method == 1:
        return svm.SVR()  # == 회귀용 svm
    elif method == 2:
        return ensemble.RandomForestRegressor()
    elif method == 3:
        return neighbors.KNeighborsRegressor()  # == k-근접법: 미지의 데이터에서 가장 가까운 학습 데이터를 k개 골라냄
    else:
        print("Error: No regression model")
        sys.exit(0)


def main():
    if len(sys.argv) < 2:
        print("Usage: python regression_models.py models")
        sys.exit(0)

    method = int(sys.argv[1])

    x, y = create_data()

    model = create_model(method)
    model.fit(x, y)

    score = model.score(x, y)
    print("R-squared: ", score)

    plt.scatter(x, y, marker='+')
    plt.scatter(x, model.predict(x), marker='o')
    plt.show()

if __name__ == "__main__":
    main()