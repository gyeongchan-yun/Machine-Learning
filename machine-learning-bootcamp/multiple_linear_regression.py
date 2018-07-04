import matplotlib.pyplot as plt
import  numpy as np
from sklearn import linear_model

# == data formation

x1 = np.random.rand(100, 1)  # 0~1 100개 난수 생성
x1 = x1 * 4 - 2  # 범위 -2 ~ 2

x2 = np.random.rand(100, 1)  # 0~1 100개 난수 생성
x2 = x2 * 4 - 2  # 범위 -2 ~ 2

y = 3 * x1 - 2 * x2 + 1  # y=3x-2

noise = np.random.randn(100, 1)
y += noise

# == Data shape transformation
x1_x2 = np.c_[x1, x2]  # [[x1_1, x2_1] , [x1_2, x2_1], ... ,[x1_100, x2_100]]

model = linear_model.LinearRegression()
model.fit(x1_x2, y)

y_predict = model.predict(x1_x2)
# print("y_predict: \n", y_predict)

score = model.score(x1_x2, y)

print("model coefficient: ", model.coef_)  # 기울기
print("model intercept: ", model.intercept_)  # 절편
print("R-squared: ", score)  # 모델의 결정 계수

plt.subplot(1, 2, 1)  # == 1 x 2 plot 에서 첫번째 sub-plot 만들기
plt.scatter(x1, y, marker='+')
plt.scatter(x1, y_predict, marker='o')
plt.xlabel('x1')
plt.ylabel('y')

plt.subplot(1, 2, 2)  # == 1 x 2 plot 에서 두번째 sub-plot 만들기
plt.scatter(x2, y, marker='+')
plt.scatter(x2, y_predict, marker='o')
plt.xlabel('x2')
plt.ylabel('y')

plt.tight_layout()

plt.show()

