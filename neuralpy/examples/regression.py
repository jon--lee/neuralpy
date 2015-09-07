import matplotlib.pyplot as plt
import neuralpy
import numpy as np

inputs = np.arange(0, 1, .0005)
actual_outputs = [ [0.2+0.4*(x**2)+0.3*np.sin(15*x)+0.05*np.cos(50*x)] for x in inputs ]
scale = np.amax(actual_outputs)



mean = np.mean(inputs)
std = np.std(inputs)

inputs_norm = (inputs - mean) / std
inputs_norm = [ [x] for x in inputs_norm ]

outputs_norm = actual_outputs / scale

training_set = zip(inputs_norm, outputs_norm)


net = neuralpy.Network([1, 100, 1])

epochs = 100
learning_rate = .1
mini_batch_size = 10

net.train(training_set, epochs, learning_rate, mini_batch_size=mini_batch_size, monitor=True)

net.show_cost()


net_outputs = [ net.forward((x - mean / std))[0] for x in inputs ]

net_outputs = [ float(net.forward([[ (x-mean) / std ]])[0]) * scale for x in inputs]
plt.plot(inputs, actual_outputs)
plt.plot(inputs, net_outputs)
plt.show()




