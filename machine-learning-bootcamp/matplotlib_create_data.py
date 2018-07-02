import numpy as np
import matplotlib.pyplot as plt


# == Definitions

x_min, x_max = -1, 1
y_min, y_max = -1, 2

SCALE = 50
TEST_RATE = 0.3

# == data formation

# reshape(-1, row) -> when we don't know column and be able to be compatible with original shape.
data_x = np.arange(x_min, x_max, 1 / float(SCALE)).reshape(-1, 1)
data_y = data_x**2
data_y_with_noise = data_y + np.random.randn(len(data_y), 1) * 0.5

# == Split data with training and testing
def split_train_test(array):
    length = len(array)
    n_train = int(length * (1 - TEST_RATE))

    indices = list(range(length))
    np.random.shuffle(indices)

    idx_test = indices[n_train:]
    idx_train = indices[:n_train]

    return sorted(array[idx_train]), sorted(array[idx_test])  # numpy array is able to indexing by list


indices = np.arange(len(data_x))
idx_train, idx_test = split_train_test(indices)

x_train, y_train = data_x[idx_train], data_y_with_noise[idx_train]

x_test, y_test = data_x[idx_test], data_y_with_noise[idx_test]

# == Figure

plt.scatter(data_x, data_y_with_noise, label='target')

plt.plot(data_x, data_y, linestyle=':', label='non noise curve')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.legend()

plt.show()



