#
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
#
# example of neuralpy2 performing regression on an y = x^2 function
# using a shallow network with a linear output activation.
#

import matplotlib.pyplot as plt
import neuralpy2
import numpy as np

inputs = np.arange(0, 10, .1)
actual_outputs = [ np.sin(x) + 1 for x in inputs ]
scale = np.amax(actual_outputs)



mean = np.mean(inputs)
std = np.std(inputs)

inputs_norm = (inputs - mean) / std
inputs_norm = [ np.reshape(x, (1, 1)) for x in inputs_norm ]

outputs_norm = actual_outputs / scale

training_set = zip(inputs_norm, outputs_norm)


net = neuralpy2.Network([1, 30, 20])
net.append("mlp", 1, "linear")

epochs = 1000
learning_rate = .001
mini_batch_size = 1

net.train(training_set, epochs, learning_rate, mini_batch_size=mini_batch_size, monitor=True)

net.show_cost()



net_outputs = [ float(net.forward([[ (x-mean) / std ]])) * scale for x in inputs]
plt.plot(inputs, actual_outputs)
plt.plot(inputs, net_outputs)
plt.show()




