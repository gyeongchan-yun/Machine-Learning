import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3 , 0.1)

y_sin = np.sin(x)

x_rand = np.random.rand(100) * 6 - 3
y_rand = np.random.rand(100) * 6 - 3

plt.figure() # plt 객체 생성

plt.subplot(1, 1, 1) # 그래프 1개로 표시함을 설정

plt.plot(x, y_sin, marker='o', markersize=5, label='line')

plt.scatter(x_rand, y_rand, label='scatter')

plt.legend() # 범례

plt.grid(True)

plt.show()
