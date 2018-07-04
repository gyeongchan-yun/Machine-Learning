import matplotlib.pyplot as plt
import  numpy as np
from sklearn import linear_model

x = np.random.rand(100, 1)  # 0~1 100개 난수 생성
x = x * 2 - 1  # 범위 -1 ~ 1

y = 4 * x**3 - 3 * x**2 + 2 * x - 1  # y=4x^3-3x^2+2x-1
y += np.random.randn(100, 1)

x_train = x[:30]
y_train = y[:30]

x_test = x[30:]
y_test = y[30:]

X_TRAIN = np.c_[x_train**6, x_train**5, x_train**4, x_train**3, x_train**2, x_train]  # 6차식으로 회귀 (원래의 선형식이 몇차식인지 모른다고 가정)

model = linear_model.Ridge()
model.fit(X_TRAIN, y_train)

score = model.score(X_TRAIN, y_train)

print("model coefficient (TRAIN): ", model.coef_)  # 기울기
print("model intercept (TRAIN): ", model.intercept_)  # 절편
print("R-squared (TRAIN): ", score)  # 모델의 결정 계수

X_TEST = np.c_[x_test**6, x_test**5, x_test**4, x_test**3, x_test**2, x_test]
score = model.score(X_TEST, y_test)
print("R-squared (TEST): ", score)  # 모델의 결정 계수

plt.subplot(2, 2, 1)
plt.scatter(x, y, marker='+')
plt.title('all')

plt.subplot(2, 2, 2)
plt.scatter(x_train, y_train, marker='+')
plt.scatter(x_train, model.predict(X_TRAIN))
plt.title('train')

plt.subplot(2, 2, 3)
plt.scatter(x_test, y_test, marker='+')
plt.scatter(x_test, model.predict(X_TEST))
plt.title('test')

plt.tight_layout()

plt.show()