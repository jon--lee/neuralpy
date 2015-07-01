import neuralpy
import numpy as np

net = neuralpy.Network([2, 3, 10, 1])

inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
outputs = [[1], [1], [1], [0]]

xLength = len(inputs[0])
yLength = len(outputs[0])

inputs = [np.reshape(x, (xLength, 1)) for x in inputs]
outputs = [np.reshape(y, (yLength, 1)) for y in outputs]

training_data = zip(inputs, outputs)
x = np.array([1, 1])
x = np.reshape(x, (2,1))

neuralpy.output(net.feedforward(x))

weights = net.weights;
biases = net.biases;

net.gradient_descent(training_data, 100, 1, 1, monitor_cost = True)

net.weights = weights;
net.biases = biases

net.gradient_descent(training_data, 100, 1, 4, monitor_cost = True)

neuralpy.output(net.feedforward(x))

net.show_costs()

