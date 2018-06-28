import numpy as np
from scipy.special import expit

class neuralNetwork:

    def __init__(self, num_input_nodes, num_hidden_nodes, num_output_nodes, learning_rate):
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes

        self.learning_rate = learning_rate

        self.weight_input_hidden = np.random.normal(0.0,
                                                    pow(self.num_hidden_nodes, -0.5),
                                                    (self.num_hidden_nodes, self.num_input_nodes)
                                                    )

        self.weight_hidden_output = np.random.normal(0.0,
                                                    pow(self.num_output_nodes, -0.5),
                                                    (self.num_output_nodes, self.num_hidden_nodes)
                                                    )

        self.activation_function = lambda x : expit(x) # sigmoid function


    def query(self, input_list):
        self.inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.weight_input_hidden, self.inputs)
        self.hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weight_hidden_output, self.hidden_outputs)
        self.final_outputs = self.activation_function(final_inputs)

        return self.final_outputs


    def train(self, input_list, target_list):
        targets = np.array(target_list, ndmin=2).T

        self.query(input_list)

        output_error = targets - self.final_outputs

        hidden_error = np.dot(self.weight_hidden_output.T, output_error)

        self.weight_hidden_output += self.learning_rate * np.dot((output_error * self.final_outputs * (1.0 - self.final_outputs)),
                                                                 np.transpose(self.hidden_outputs)
                                                                 )

        self.weight_input_hidden += self.learning_rate * np.dot((hidden_error * self.hidden_outputs * (1.0 - self.hidden_outputs)),
                                                                np.transpose(self.inputs)
                                                                )

