from neural_network import neuralNetwork

import numpy as np

num_input_nodes = 784
num_hidden_nodes = 200  # random value -> we can choose the best value by heuristic
num_output_nodes = 10  # label is between 0 ~ 9
learning_rate = 0.1  # random value -> we can choose the best value by heuristic

neural_network = neuralNetwork(num_input_nodes, num_hidden_nodes, num_output_nodes, learning_rate)


def training():
    training_data_file = open("../mnist_dataset/mnist_train_100.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 5  # random value -> we can choose the best value by heuristic

    print("Start training!")
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # adjust input value to between 0.01 and 0.99

            targets = np.zeros(num_output_nodes) + 0.01  # To avoid 0

            targets[int(all_values[0])] = 0.99  # answer target

            neural_network.train(inputs, targets)

    print("Training is finished.\n")


def test():
    test_data_file = open("../mnist_dataset/mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    score_list = []

    print("Start Testing!")
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print("\ncorrect label: ", correct_label)

        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # adjust input value to between 0.01 and 0.99

        outputs = neural_network.query(inputs)
        print("Neural Network's outputs: \n", outputs)

        output_label = np.argmax(outputs)
        print("Neural Network's answer: ", output_label)

        if output_label == correct_label:
            score_list.append(1)
        else:
            score_list.append(0)

    print("\nTesting is finished.")

    print("\nScore List: ", score_list)
    score_array = np.asarray(score_list)
    performance = score_array.sum() / score_array.size
    print("Performance = ", performance)


if __name__ == "__main__":
    training()
    test()