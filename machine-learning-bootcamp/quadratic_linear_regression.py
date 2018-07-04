import matplotlib.pyplot as plt
import  numpy as np
from sklearn import linear_model

x = np.random.rand(100, 1)  # 0~1 100개 난수 생성
x = x * 4 - 2  # 범위 -2 ~ 2

y = 3 * x**2 - 2  # y=3x^2-2
y += np.random.randn(100, 1)  # 표준 정규분포에서의 100개의 난수 더하기 (noise)

model = linear_model.LinearRegression()
model.fit(x**2, y)
score = model.score(x**2, y)

print("model coefficient: ", model.coef_)  # 기울기
print("model intercept: ", model.intercept_)  # 절편
print("R-squared: ", score)  # 모델의 결정 계수

plt.scatter(x, y, marker='+')
plt.scatter(x, model.predict(x**2), marker='o')
plt.show()