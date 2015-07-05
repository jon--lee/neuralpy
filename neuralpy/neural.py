import numpy as np
import random
import matplotlib.pyplot as plt
import logging

def output(x = ""):
	print bcolors.WARNING + str(x) + bcolors.ENDC



class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def cost_function(output, y):
	return 0.5 * ((output - y) ** 2)

def cost_derivative(output, y):
	return output - y


def sigmoid(x):
	return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
	sig = sigmoid(x)
	return sig * ( 1- sig )


# converting a list object to a column vector (or n x 1 matrix)
def list2vec(li):
	li = np.array(li)
	return np.reshape(li, (len(li), 1))


class errors:
	init_args_general_error = '__init__() requires two or more non-zero integers separated by commas or a list of two or more non-zero integers.'




class Network():
	
	def __init__(self, *args):

		# handle ambiguous arguments, raise an error if not passed
		try:
			layers = self.handle_init_args(args)
		except TypeError:
			raise TypeError(errors.init_args_general_error)

		self.num_layers = len(layers)

		self.cost_plots = []															# list of lists that correspond to the y-axis of the cost function plot

		self.layers = layers
		self.randomize_parameters()
		# self.biases = [np.random.randn(y, 1) for y in layers[1:]]
		# self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

	# method: handle_init_args
	# handles arguments passed to initializer as they can be ambiguous
	# could be a list or a tuple. Must handle code accordingly
	def handle_init_args(self, args):
		if len(args) == 1: layers = [abs(layer) for layer in args[0]]
		else: layers = [abs(layer) for layer in list(args)]

		if self.are_valid_layers(layers):
			return layers
		else:
			raise TypeError()


	# method: are_valid_layers
	# determines if the given list of layer lengths
	# are actually valid (that is, they are greater than 0 and the length
	# of the list is greater than 1).
	def are_valid_layers(self, layers):
		if len(layers) < 2: return False
		for layer in layers:
			if layer <= 0: return False
		return True
		

	def feedforward(self, a):
		a = list2vec(a)
		for b, w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w, a)+b)
		return np.reshape(a, (len(a))).tolist()

	def show_costs(self):
		for plot in self.cost_plots:
			x = xrange(0, len(plot), 1)
			plt.plot(x, plot)
		plt.show()

	def clear_costs(self):
		self.cost_plots = []

	def randomize_parameters(self):
		self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]

	def train(self, training_data, epochs, eta, batch_length = 0, monitor_cost = False):
		training_data = [ (list2vec(pair[0]), list2vec(pair[1])) for pair in training_data ]
		n = len(training_data)

		eta = float(eta)
		if(batch_length <= 0):
			batch_length = n
		if monitor_cost:
			self.cost_plots.append([])					# if were tracking the cost, create a new plot to track this descent
			num_cost_plots = len(self.cost_plots)		# number of cost plots that we have so far

		# iterate over the number of epochs and update every weight/bias for all training sets in the batch
		for j in xrange(epochs):
			random.shuffle(training_data)
			batch = training_data[:batch_length]
			self.update_batch(training_data, eta)

			# if there is a request to monitor the cost, we want to monitor the cost after 
			# the update form each epoch. Rather than recalculating the feed forward, it would be better
			# to have that data in a dictionary
			if(monitor_cost):
				cost_components = []
				for x, y in batch:
					output = self.feedforward(x)
					cost_components.append(cost_function(output, y))
				
				cost = np.linalg.norm(cost_components)
				self.cost_plots[num_cost_plots - 1].append(cost)


	gradient_descent = train 				# this is just for legacy issues


	def update_batch(self, batch, eta):
		delta_biases = [np.zeros(b.shape) for b in self.biases]
		delta_weights = [np.zeros(w.shape) for w in self.weights]
		
		for x, y in batch:
			partial_biases, partial_weights = self.backpropagation(x, y)

			delta_biases = [db+pb for db, pb in zip(delta_biases, partial_biases)]
			delta_weights = [dw+pw for dw, pw in zip(delta_weights, partial_weights)]



		self.weights = [w-(eta)*dw for w, dw in zip(self.weights, delta_weights)]
		self.biases = [b-(eta)*nb for b, nb in zip(self.biases, delta_biases)]



	def backpropagation(self, x, y):
		partial_biases = [np.zeros(b.shape) for b in self.biases]
		partial_weights = [np.zeros(w.shape) for w in self.weights]

		# feedforward
		activation = x
		activations = [x] 	# list to store all the activations, layer by layer
		sums = [] 			# list to store all the u vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			u = np.dot(w, activation)+b
			sums.append(u)
			activation = sigmoid(u)
			activations.append(activation)

		# backward pass
		# sigmoid prime vector can be optimized as it is just an equation dependent on the components
		# of the activation layer (the sigmoid part has already been computed, no need to compute it again)
		epsilon = cost_derivative(activations[-1], y) * sigmoid_prime(sums[-1])
		partial_biases[-1] = epsilon
		partial_weights[-1] = np.dot(epsilon, activations[-2].transpose())

		for l in xrange(2, self.num_layers):
			u = sums[-l]
			spv = sigmoid_prime(u)
			epsilon = np.dot(self.weights[-l+1].transpose(), epsilon) * spv
			partial_biases[-l] = epsilon
			partial_weights[-l] = np.dot(epsilon, activations[-l-1].transpose())
		return (partial_biases, partial_weights)










