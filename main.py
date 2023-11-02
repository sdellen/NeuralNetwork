import numpy as np
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist

foo = 1

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_prime(x): return x * (1 - x)
def tanh(x): return np.tanh(x)
def tanh_prime(x): return 1 - x ** 2
def relu(x): return np.max(0, x)
def relu_prime(x): return 1 if x > 0 else 0

class Network():
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.layers: list[Layer] = []
        self.loss = 0

    def add_layer(self, num_nodes, activation=sigmoid, activation_prime=sigmoid_prime):
        num_next_inputs = self.num_inputs if len(self.layers) == 0 else self.layers[-1].num_nodes
        layer = Layer(num_next_inputs, num_nodes, activation, activation_prime)
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, inputs, target, learning_rate):
        self.output = self.forward(inputs)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            if layer == self.layers[-1]:
                layer.calculate_delta(target)
            else:
                next_layer = self.layers[i + 1]
                layer.calculate_delta_next(next_layer.deltas, next_layer.W)

            if i > 0:
                layer.backward(self.layers[i - 1].a, learning_rate)
            else:
                layer.backward(inputs, learning_rate)

    def train(self, inputs, outputs, epochs, learning_rate):
        self.loss = []
        start_time = time.time()  # start the timer
        epoch_loss = 0

        for i in range(epochs):
            epoch_loss = 0
            for j in range(len(inputs)):
                self.backward(inputs[j], outputs[j], learning_rate)
                epoch_loss += np.mean(np.square(outputs[j] - self.output))

            epoch_loss /= len(inputs)
            self.loss.append(epoch_loss)

            # print progress and estimated time to completion
            if i % (epochs//100) == 0:
                progress = int((i / epochs) * 100)
                bar = ('=' * (int(progress/5)-1) + '>') if foo else ('8=' + '=' * (int(progress/5)-3) + 'D')
                print(f"Progress: [{bar:<20}] {progress:>3}% Loss: {epoch_loss:.5e}", end="\r", flush=True)

        # print final progress and elapsed time
        print(f"Progress: [{'=' * 20}] 100% Loss: {epoch_loss:.5e}", end="\n", flush=True) if foo else print(f"Progress: [{'8' + '=' * 18 + 'D'}] 100% Loss: {epoch_loss:.5e}", end="\n", flush=True)
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")  # print elapsed time

class Layer():
    def __init__(self, input_size, output_size, activation=sigmoid, activation_prime=sigmoid_prime):
        self.input_size = input_size
        self.num_nodes = output_size
        self.activation = activation
        self.activation_prime = activation_prime
        self.deltas = np.zeros((1, output_size))
        
        self.W = np.random.randn(self.input_size, self.num_nodes)
        self.b = np.random.randn(self.num_nodes)
        
    def forward(self, inputs):
        self.z = np.dot(inputs, self.W) + self.b
        self.a = self.activation(self.z)
        return self.a
    
    def calculate_delta(self, target):
        self.deltas = (self.a - target) * self.activation_prime(self.a)

    def calculate_delta_next(self, next_delta, next_W):
        self.deltas = np.dot(next_delta, next_W.T) * self.activation_prime(self.a)
    
    def backward(self, inputs, learning_rate):
        self.b -= learning_rate * self.deltas
        self.W -= learning_rate * self.deltas * inputs[:, None]

def main():
    # hyperparameters
    num_samples = 100
    num_epochs = 100
    learning_rate = 1

    # load the mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape and normalize the images
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_train = x_train.astype('float32') / 255
    y_train = np.eye(10)[y_train]
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = x_test.astype('float32') / 255
    y_test = np.eye(10)[y_test]

    # only use a subset of the data to train
    inputs_train = x_train[:num_samples]
    outputs_train = y_train[:num_samples]

    # create a network
    net = Network(784) # 28x28 images. this is the inputs to the network
    net.add_layer(16) # hidden layer with 16 nodes
    net.add_layer(16) # hidden layer with 16 nodes
    net.add_layer(10) # output layer with 10 nodes

    # train the network
    net.train(inputs_train, outputs_train, num_epochs, learning_rate)

    # test the network
    inputs_test = x_test[:60000]
    outputs_test = y_test[:60000]

    # calculate and print the accuracy
    incorrect_predictions = 0
    for i in range(len(inputs_test)):
        incorrect_predictions += np.argmax(net.forward(inputs_test[i])) == np.argmax(outputs_test[i])
    print(f"Accuracy: {incorrect_predictions/len(inputs_test)*100:.2f}%")

    # plot the loss over time, aka epochs
    plt.plot(net.loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main()