import numpy as np
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist
import json
import os
foo = 1


def sig(x): return 1 / (1 + np.exp(-x))
def sig_prime(x): return x * (1 - x)
def tanh(x): return np.tanh(x)
def tanh_prime(x): return 1 - x ** 2
def relu(x): return np.max(0, x)
def relu_prime(x): return 1 if x > 0 else 0


class Network():
  def __init__(self, num_inputs):
    self.num_inputs = num_inputs
    self.layers: list[Layer] = []
    self.loss = 0

  def add_layer(self, num_nodes, activation=sig, activation_prime=sig_prime):
    num_next_inputs = self.num_inputs if len(self.layers) == 0 else self.layers[-1].num_nodes
    layer = Layer(num_next_inputs, num_nodes, activation, activation_prime)
    self.layers.append(layer)
    return self

  def forward(self, inputs):
    for layer in self.layers:
      inputs = layer.forward(inputs)
    return inputs

  def backward(self, inputs, target, lr):
    self.output = self.forward(inputs)

    self.layers[-1].calc_delta(target)
    for i in range(len(self.layers)-2, -1,-1):
      self.layers[i].calc_delta_next(self.layers[i+1].deltas, self.layers[i+1].W)

    input = inputs
    for layer in self.layers:
      input = layer.backward(input, lr).a

  def train(self, inputs, outputs, epochs, learning_rate):
    self.loss = []
    start_time = time.time()  # start the timer
    epoch_loss = 0

    for i in range(epochs):
      epoch_loss = 0
      for j in range(len(inputs)):
        self.backward(inputs[j], outputs[j], learning_rate)
        epoch_loss += 0.5*np.mean(np.square(outputs[j] - self.output))

      epoch_loss /= len(inputs)
      self.loss.append(epoch_loss)

      # print progress
      if i % (max(epochs,100)//100) == 0:
        progress = int((i / epochs) * 100)
        bar = ('=' * (int(progress/5)-1) + '>') if foo else ('8=' + '=' * (int(progress/5)-3) + 'D')
        print(f"Progress: [{bar:<20}] {progress:>3}% Loss: {epoch_loss:.5e}", end="\r", flush=True)

    # print final progress and elapsed time
    print(f"Progress: [{'=' * 20}] 100% Loss: {epoch_loss:.5e}", end="\n", flush=True) if foo else print(f"Progress: [{'8' + '=' * 18 + 'D'}] 100% Loss: {epoch_loss:.5e}", end="\n", flush=True)
    elapsed_time = time.time() - start_time
    # print elapsed time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return self.loss
  # export the weights and biases to json file
  def export(self, filename):
    
    data = []
    for layer in self.layers:
      W, b = layer.get_params()
      data.append({"W": W.tolist(), "b": b.tolist()})
    with open(filename, "w") as f:
      json.dump(data, f)

  # import the weights and biases from json file
  def import_(self, filename):
    with open(filename, "r") as f:
      data = json.load(f)
    for i, layer in enumerate(self.layers):
      layer.W = np.array(data[i]["W"])
      layer.b = np.array(data[i]["b"])


class Layer():
  def __init__(self, input_size, output_size, activation=sig, activation_prime=sig_prime):
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

  def calc_delta(self, target):
    self.deltas = (self.a - target) * self.activation_prime(self.a)
    return self

  def calc_delta_next(self, next_delta, next_W):
    self.deltas = np.dot(next_delta, next_W.T) * self.activation_prime(self.a)
    return self

  def backward(self, inputs, learning_rate):
    self.b -= learning_rate * self.deltas
    self.W -= learning_rate * self.deltas * inputs[:, None]
    return self
  
  # get the weights and biases
  def get_params(self):
    return self.W, self.b

def main_train():
  num_train_samples,num_epochs,learning_rate = 600,50,1 # hyperparameters
  (x_train, y_train), (x_test, y_test) = mnist.load_data() # load the mnist dataset

  # reshape and normalize the images
  x_train,y_train = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255,np.eye(10)[y_train]
  x_test,y_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255,np.eye(10)[y_test]

  # only use a subset of the data to train
  inputs_train,outputs_train = x_train[:num_train_samples],y_train[:num_train_samples]

  net = Network(784).add_layer(5).add_layer(10) # initializes the network
  loss = net.train(inputs_train, outputs_train, num_epochs, learning_rate) # train the network

  # calculate and print the accuracy
  corrects = 0
  for i in range(len(x_test)):
    corrects += (np.argmax(net.forward(x_test[i])) == np.argmax(y_test[i]))
  print(f"Accuracy: {corrects/len(x_test)*100:.2f}%")

  # export the weights and biases to json file
  name = f"{net.num_inputs}"
  for layer in net.layers:
    name += f"-{layer.num_nodes}"
  
  # check if file exists, if so, append a number to the end
  if os.path.exists(f"network-{name}.json"):
    i = 1
    while os.path.exists(f"network-{name}-({i}).json"): i+=1
    name += f"-({i})"
      

  net.export(f"network-{name}.json")

  # export the loss to csv file
  with open(f"loss-{name}.csv", "w") as f:
    f.write("epoch,loss\n")
    for i, loss in enumerate(loss):
      f.write(f"{i},{loss}\n")
  

  # plot the loss over time, aka epochs
  plt.plot(net.loss)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()


def main_test():
  (_, _), (x_test, y_test) = mnist.load_data() # load the mnist dataset
  x_test,y_test = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255,np.eye(10)[y_test]

  net = Network(784).add_layer(32).add_layer(10) # initializes the 784->16->16->10 network
  net.import_("network-784-32-10.json") # import the weights and biases from json file

  # calculate and print the accuracy
  corrects = 0
  for i in range(len(x_test)):
    corrects += (np.argmax(net.forward(x_test[i])) == np.argmax(y_test[i]))
  print(f"Accuracy: {corrects/len(x_test)*100:.2f}%")


main_train()