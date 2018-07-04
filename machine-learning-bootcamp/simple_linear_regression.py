import matplotlib.pyplot as plt
import  numpy as np
from sklearn import linear_model

import sys


def create_data(noise):
    x = np.random.rand(100, 1)  # 0~1 100개 난수 생성
    x = x * 4 - 2  # 범위 -2 ~ 2
    y = 3 * x - 2  # y=3x-2

    if noise:
        y += np.random.randn(100, 1)  # 표준 정규분포에서의 100개의 난수 더하기 (noise)

    return x, y


def main():
    if len(sys.argv) < 2:
        print("Usage: python simple_linear_regression.py noise(0 or 1)")
        sys.exit(0)

    noise = True if int(sys.argv[1]) == 1 else False

    x, y = create_data(noise)

    model = linear_model.LinearRegression()
    model.fit(x, y)

    print("model coefficient: ", model.coef_)  # 기울기
    print("model intercept: ", model.intercept_)  # 절편

    plt.scatter(x, y, marker='+')
    plt.scatter(x, model.predict(x), marker='o')
    plt.show()

if __name__ == "__main__":
    main()