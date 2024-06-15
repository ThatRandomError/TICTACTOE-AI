import pickle

import numpy as np


class NN:
    def __init__(self, layers):
        self.layers = layers
        self.output = []
        self.dataset = []
        self.nn = []
        self.initialize_gradients()

    def load_dataset(self, dataset):
        self.dataset = dataset

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __cost(self, output, target):
        return np.sum((output - target) ** 2)

    def initialize_gradients(self):
        self.gradients = []
        for i in range(len(self.layers) - 1):
            self.gradients.append([
                np.zeros((self.layers[i], self.layers[i + 1])), 
                np.zeros(self.layers[i + 1])
            ])

    def __update_values(self, learnRate):
        for index, (weights, biases) in enumerate(self.nn):
            weights -= self.gradients[index][0] * learnRate
            biases -= self.gradients[index][1] * learnRate

    def forward_pass(self, inputs):
        activations = [np.array(inputs)]
        zs = []
        for weights, biases in self.nn:
            z = np.dot(activations[-1], weights) + biases
            zs.append(z)
            activation = self.__sigmoid(z)
            activations.append(activation)
        self.output = activations[-1]
        return activations, zs

    def train(self, epochs, learnRate, printdata=False):
        for epoch in range(epochs):
            total_cost = 0
            for data in self.dataset:
                inputs, target = data

                activations, zs = self.forward_pass(inputs)

                cost = self.__cost(activations[-1], target)
                total_cost += cost

                delta = (activations[-1] - target) * self.__sigmoid_derivative(activations[-1])
                self.gradients[-1][0] = np.outer(activations[-2], delta)
                self.gradients[-1][1] = delta

                for l in range(2, len(self.layers)):
                    z = zs[-l]
                    sp = self.__sigmoid_derivative(activations[-l])
                    delta = np.dot(delta, self.nn[-l + 1][0].T) * sp
                    self.gradients[-l][0] = np.outer(activations[-l - 1], delta)
                    self.gradients[-l][1] = delta

                self.__update_values(learnRate)

            if printdata:
                print(f"Epoch: {epoch}, Cost: {total_cost / len(self.dataset)}")

    def prediction(self, inputs):
        activations = np.array(inputs)
        for weights, biases in self.nn:
            activations = self.__sigmoid(np.dot(activations, weights) + biases)
        self.output = activations
        return self.output

    def randomize_values(self):
        self.nn = []
        for i in range(len(self.layers) - 1):
            self.nn.append([
                np.random.randn(self.layers[i], self.layers[i + 1]), 
                np.random.randn(self.layers[i + 1])
            ])

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self.nn, f)

    def load(self, name):
        with open(name, 'rb') as f:
            self.nn = pickle.load(f)

    def print_output(self):
        print(self.output)

    def get_output(self):
        return self.output

    def accuracy(self):
        correct_predictions = 0
        total_predictions = 0
        for i in self.dataset:
            self.prediction(i[0])
            pos1 = 0
            highest = 0
            for index, j in enumerate(self.output):
                if j > highest:
                    highest = j
                    pos1 = index
            pos2 = 0
            highest = 0
            for index, j in enumerate(i[1]):
                if j > highest:
                    highest = j
                    pos2 = index
            print(f"exxpected: {i[1][pos2]}")
            print(f"accutally: {self.output[pos1]}")
            if pos2 == pos1:
                correct_predictions += 1
                print("Test")
            total_predictions += 1
        print(f"accuracy: {(correct_predictions/total_predictions) * 100}")