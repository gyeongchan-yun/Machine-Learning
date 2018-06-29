import numpy as np
import matplotlib.pyplot as plt

import sys


def show_image(data_list, index):
    # First element of all_values is handwriting value (label)
    all_values = data_list[index].split(',')  # length is 785. 28 x 28 pixel size.

    # == asfarray -> change string to number and store it as array == #
    image_array = np.asfarray(all_values[1:]).reshape((28, 28))

    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python handwriting_image.py index")
        sys.exit(0)

    index = int(sys.argv[1])

    data_file = open("../mnist_dataset/mnist_train_100.csv", 'r')
    data_list = data_file.readlines()  # readlines() -> memory overhead for huge file
    data_file.close()

    show_image(data_list, index)


if __name__ == "__main__":
    main()