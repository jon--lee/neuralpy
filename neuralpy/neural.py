import numpy as np
import random
import matplotlib.pyplot as plt

def output(x = ""):
	print bcolors.OKGREEN + str(x) + bcolors.ENDC

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

cost_function_vec = np.vectorize(cost_function)
cost_derivative_vec = np.vectorize(cost_derivative)

def sigmoid(x):
	x = float(x)
	return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(x):
	sig = sigmoid(x)
	return sig * ( 1- sig )

sigmoid_vec = np.vectorize(sigmoid)
sigmoid_prime_vec = np.vectorize(sigmoid_prime)


class Network():
	
	def __init__(self, sizes):
		self.num_layers = len(sizes)

		self.cost_plots = []															# list of lists that correspond to the y-axis of the cost function plot

		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			a = sigmoid_vec(np.dot(w, a)+b)
		return a

	def show_costs(self):
		x = xrange(0, len(self.cost_plots[0]), 1)
		for plot in self.cost_plots:
			plt.plot(x, plot)
		plt.show()

	def gradient_descent(self, training_data, epochs, eta, mini_batch_size = 0, monitor_cost = False):
		n = len(training_data)
		eta = float(eta)
		if(mini_batch_size <= 0):
			mini_batch_size = n
		if monitor_cost:
			self.cost_plots.append([])
			num_cost_plots = len(self.cost_plots)

		for j in xrange(epochs):
			random.shuffle(training_data)
			batch = training_data[:mini_batch_size]
			self.update_batch(training_data, eta)

			if(monitor_cost):
				cost_components = []
				for x, y in batch:
					output = self.feedforward(x)
					cost_components.append(cost_function_vec(output, y))
				
				cost = np.linalg.norm(cost_components)
				self.cost_plots[num_cost_plots - 1].append(cost)

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
		activations = [x] # list to store all the activations, layer by layer
		sums = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			sums.append(z)
			activation = sigmoid_vec(z)
			activations.append(activation)

		# backward pass
		# sigmoid prime vector can be optimized as it is just an equation dependent on the components
		# of the activation layer (the sigmoid part has already been computed, no need to compute it again)
		delta = cost_derivative(activations[-1], y) * sigmoid_prime_vec(sums[-1])
		partial_biases[-1] = delta
		partial_weights[-1] = np.dot(delta, activations[-2].transpose())

		for l in xrange(2, self.num_layers):
			z = sums[-l]
			spv = sigmoid_prime_vec(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
			partial_biases[-l] = delta
			partial_weights[-l] = np.dot(delta, activations[-l-1].transpose())
		return (partial_biases, partial_weights)










