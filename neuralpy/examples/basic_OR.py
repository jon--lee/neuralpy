# 
# @author Jonathan N. Lee
# 
# The purpose of this example is to demonstrate the usage
# of neuralpy and create a basic network that is able to
# represent the very basic OR function. Here is a truth
# table depicting what OR should achieve:
# 	
# 	_____________________________
# 	|		|		|			|
# 	|	A 	|	B 	| 	output 	|
# 	|_______|_______|___________|
# 	|	T 	|	T 	|	  T 	|
# 	|_______|_______|___________|
# 	|	F 	|	T 	|	  T 	|
# 	|_______|_______|___________|
# 	|	T 	|	F 	|	  T 	|
# 	|_______|_______|___________|
# 	|	F 	|	F 	|	  F 	|
# 	|_______|_______|___________|
# 	
# 	In our network, 1 will represent True and
# 	0 will represent False
# 
import neuralpy

# set up a basic neural network with a 2-node input layer, 
# one 3-neuron hidden layer, and one 1-neuron output layer
net = neuralpy.Network(2, 3, 1)

# here is some arbitrary input that we will use to test
x = [1, 1]
out = net.feedforward(x)
neuralpy.output(out)

# here is our training_data the reflects the truth table
# in the header of this file
datum_1 = ([1, 1], [1])
datum_2 = ([1, 0], [1])
datum_3 = ([0, 1], [1])
datum_4 = ([0, 0], [0])

training_data = [datum_1, datum_2, datum_3, datum_4]

# we set our other hyperparameter and the number of
# epochs that we want to train the network for
learning_rate = 1
epochs = 100

# train the network
net.train(training_data, epochs, learning_rate)

# see if it actually worked!
out = net.feedforward(x)
neuralpy.output(out)