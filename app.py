#
# Learning neural networks from scratch
#

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.inputs = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def __init__(self):
        self.output = None
        self.inputs = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


#
# Softmax non-linear activation function
#
class Activation_Softmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def __init__(self):
        self.dinputs = None

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    # Forward pass
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# X, y = spiral_data(samples=100, classes=3)
#
# dense1 = Layer_Dense(2, 3)
# activation1 = Activation_ReLU()
# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()
# loss_function = Loss_CategoricalCrossentropy()
#
# dense1.forward(X)
# activation1.forward(dense1.output)
# dense2.forward(activation1.output)
# activation2.forward(dense2.output)
# print(activation2.output[:5])
# loss = loss_function.calculate(activation2.output, y)
# print('Loss: ', loss)
# predictions = np.argmax(activation2.output, axis=1)
# if len(y.shape) == 2:
#     y = np.argmax(y, axis=1)
# accuracy = np.mean(predictions == y)
# print('Acc:', accuracy)

# softmax_output = np.array([[0.7, 0.1, 0.2],
#                            [0.1, 0.5, 0.5],
#                            [0.02, 0.9, 0.08]])
# class_targets = np.array([0, 1, 1])
# sofmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
# sofmax_loss.backward(softmax_output, class_targets)
# dvalues1 = sofmax_loss.dinputs
#
# activation = Activation_Softmax()
# activation.output = softmax_output
# loss = Loss_CategoricalCrossentropy()
# loss.backward(softmax_output, class_targets)
# activation.backward(loss.dinputs)
# dvalues2 = activation.dinputs
#
# print('Gradients: combined loss and activation:')
# print(dvalues1)
# print('Gradients: separate loss and activation:')
# print(dvalues2)

# X, y = spiral_data(samples=100, classes=3)
#
# dense1 = Layer_Dense(2, 3)
# activation1 = Activation_ReLU()
# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()
#
# loss_function = Loss_CategoricalCrossentropy()
#
# lowest_loss = 9999999
#
# best_dense1_weights = dense1.weights.copy()
# best_dense1_biases = dense1.biases.copy()
# best_dense2_weights = dense2.weights.copy()
# best_dense2_biases = dense2.biases.copy()
#
# for iteration in range(10000):
#     dense1.weights = 0.05 * np.random.randn(2, 3)
#     dense1.biases = 0.05 * np.random.randn(1, 3)
#
#     dense2.weights = 0.05 * np.random.randn(3, 3)
#     dense2.biases = 0.05 * np.random.randn(1, 3)
#
#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     activation2.forward(dense2.output)
#
#     loss = loss_function.calculate(activation2.output, y)
#
#     predictions = np.argmax(activation2.output, axis=1)
#     accuracy = np.mean(predictions == y)
#
#     if (loss < lowest_loss):
#         print("New set of weights found, interation:", iteration,
#               "loss:", loss, 'acc:', accuracy)
#         best_dense1_weights = dense1.weights.copy()
#         best_dense1_biases = dense1.biases.copy()
#
#         best_dense2_weights = dense2.weights.copy()
#         best_dense2_biases = dense2.biases.copy()
#         lowest_loss = loss
# Create dataset
# X, y = spiral_data(samples=100, classes=3)
# # Create Dense layer with 2 input features and 3 output values
# dense1 = Layer_Dense(2, 3)
# # Create ReLU activation (to be used with Dense layer):
# activation1 = Activation_ReLU()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()
#
# loss_function = Loss_CategoricalCrossentropy()
#
# lowest_loss = 9999999
#
# best_dense1_weights = dense1.weights.copy()
# best_dense1_biases = dense1.biases.copy()
# best_dense2_weights = dense2.weights.copy()
# best_dense2_biases = dense2.biases.copy()
#
# for iteration in range(10000):
#     dense1.weights = 0.05 * np.random.randn(2, 3)
#     dense1.biases = 0.05 * np.random.randn(1, 3)
#
#     dense2.weights = 0.05 * np.random.randn(3, 3)
#     dense2.biases = 0.05 * np.random.randn(1, 3)
#
#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)
#     activation2.forward(dense2.output)
#
#     loss = loss_function.calculate(activation2.output, y)
#
#     predictions = np.argmax(activation2.output, axis=1)
#     accuracy = np.mean(predictions == y)
#
#     if (loss < lowest_loss):
#         print("New set of weights found, interation:", iteration,
#               "loss:", loss, 'acc:', accuracy)
#         best_dense1_weights = dense1.weights.copy()
#         best_dense1_biases = dense1.biases.copy()
#
#         best_dense2_weights = dense2.weights.copy()
#         best_dense2_biases = dense2.biases.copy()
#         lowest_loss = loss
# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(3, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)
# Let's see output of the first few samples:
print(loss_activation.output[:5])
# Print loss value
print('loss:', loss)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
# Print accuracy
print('acc:', accuracy)
# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)
