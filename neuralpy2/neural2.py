#
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# NetworkBase module that defines an abstract network. sizes parameter determines
# the number of nodes in each respective layer. layers by default
# consist of an inputs and mlps with sigmoid activations.
# 


# system libraries
import random
# internal libraries
import layers
import activations
import colors
import costs
# third party libraries
import numpy as np
import matplotlib.pyplot as plt

def output(s=""):
    print colors.green + str(s) + colors.end


class NetworkBase(object):

    # initialize the network by creating
    # a layers with the corresponding number of nodes
    # @params sizes     list of positive integers representing
    #                   nodes per layer
    def __init__(self, sizes):
        raise NotImplementedError


    # propagate forward by iterating over all layers
    # staring with the first layer
    # @param x          input column vector
    def forward(self, x):
        raise NotImplementedError

    # create and append layer to the end layer of the network
    # simply by using the end instance variable
    # @param type_      layer identifier type
    # @param size       number of nodes in the layer
    # @param activ      activation identifier type
    def append(self, *args):
        raise NotImplementedError

    # pop the last layer off the network
    # by setting end instance var to the 
    # previous layer and removing the previous
    # layers forward propagation to this
    def pop(self):
        raise NotImplementedError

    # train neural network based on training data
    # to optimize the weights. Abstract method
    # intended to be handled by implementing network
    # class
    # @param training_set       training_set which is a list
    #                           of tuples: first component is column
    #                           input vec, second is column output vec
    # @param epochs             number of iterations to update weights
    # @param alpha              learning rate
    # @param mini_batch_size    by default, reverts to 1
    # @param monitor            by default, no loss surveillence
    def train(self, *args):
        raise NotImplementedError

    # iterate through layer and update the biases
    # and weights as each layer type should handle
    # the individual implementation uniquely
    # applying the delta updates should always zero
    # out the delta weights
    # @param alpha              learning rate
    def _apply_updates(self, alpha):
        raise NotImplementedError

    # iterate through each alyer and zero out
    # the weights, as handled by the implementations
    # of the given layers
    def _zero_deltas(self):
        raise NotImplementedError

    # iterate through each layer and save the weights
    # and biases to a specified text file in the file_path
    # will also store the network hyperparameters
    # @param file_path      path to the write file
    def save(self, file_path):
        raise NotImplementedError

    # load weights and biases from a specified file path
    # Note: this overwrites the current network structure
    # and weights and biases. Use carefully.
    # @param file_path      path to the read file
    def load(self, file_path):
        raise NotImplementedError

    # compute the cost given a 
    def _compute_cost(self, training_set):
        raise NotImplementedError


# Network class provides the implementation of NetworkBase's 
# non implemented functions. See interface class for details about
# each funciton (no comments in this class provided)
class Network(NetworkBase):


    def __init__(self, sizes):
        it = iter(sizes)
        self.start = layers.Input(next(it))              
        layer = self.start
        for size in it:
            layer.append( layers.MLP(size, activations.sigmoid) )
            layer = layer.next_
        self.end = layer


    def forward(self, x):
        x = np.array(x)
        it = iter(self.start)
        for layer in it:
            x = layer.forward(x)
        return x


    def append(self, type_, size, activ):
        type_ = layers.mapping[type_]
        activ = activations.mapping[activ]
        layer = type_(size, activ)
        self.end.append(layer)
        self.end = self.end.next_


    def pop(self):
        self.end = self.end.prev
        self.end.append(None)


    def train(self, training_set, epochs, alpha, mini_batch_size=1, monitor=False):
        self.costcurve = []
        alpha = float(alpha)
        n = len(training_set)
        self._zero_deltas()
        for j in xrange(epochs):
            random.shuffle(training_set)
            mini_batches = [
                training_set[k:k + mini_batch_size] 
                for k in xrange(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self._update_batch(mini_batch, alpha)
            if monitor:
                self.costcurve.append(self._compute_cost(training_set))

    def show_cost(self):
        x = xrange(0, len(self.costcurve), 1)
        plt.plot(x, self.costcurve)
        plt.show()


    def _update_batch(self, training_set, alpha):
        for x, y in training_set:
            self.forward(x)
            mu = costs.mean_square.deriv(self.end.x, y) *\
                self.end.activ.deriv(self.end.z)
            root = self.end
            while root is not None:
                mu = root._backward(mu)
                root = root.prev
        self._apply_updates(alpha/len(training_set))        # for testing only, confirm for production


    def _apply_updates(self, alpha):
        it = iter(self.start)
        for layer in it:
            layer._apply_updates(alpha)
            layer._zero_deltas()


    def _zero_deltas(self):
        it = iter(self.start)
        for layer in it:
            layer._zero_deltas()


    def _compute_cost(self, training_set):
        cost = 0
        for x, y in training_set:
            actual = self.forward(x)
            cost += np.linalg.norm(costs.mean_square.func(actual, y))
        return cost / len(training_set)
