import numpy as np
import time
import matplotlib.pyplot as plt

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


            if i % (epochs//100) == 0:
                progress = int((i / epochs) * 100)
                bar = ('=' * (int(progress/5)-1) + '>') if foo else ('8=' + '=' * (int(progress/5)-3) + 'D')
                print(f"Progress: [{bar:<20}] {progress:>3}% Loss: {epoch_loss:.5e} ", end="\r", flush=True)

        print(f"Progress: [{'=' * 20}] 100% Loss: {epoch_loss:.5e}", end="\n", flush=True) if foo else print(f"Progress: [{'8' + '=' * 18 + 'D'}] 100% Loss: {epoch_loss:.5e}", end="\n", flush=True)

        end_time = time.time()  # end the timer
        elapsed_time = end_time - start_time  # calculate elapsed time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")  # print elapsed time

    def predict(self, input):
        return self.forward(input)

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
    # create a dataset to train a network for the sum operation
    # inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # outputs = np.array([[0], [1], [1], [0]])

    inputs = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    outputs = np.array([[int(1)], [0.2], [0.3]])

    # create a network
    net = Network(3)
    net.add_layer(10)
    net.add_layer(1)

    # train the network
    net.train(inputs, outputs, 10000, 1)

    # test the network
    for i in range(len(inputs)):
        print(f"Output: {net.predict(inputs[i])[0]:.5f} Target: {outputs[i][0]:.5f}")

    # # plot the loss over time
    # plt.plot(net.loss)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()

if __name__ == "__main__":
    main()