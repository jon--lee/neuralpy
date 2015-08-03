# 
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# layer defines a interface layer and related functions such
# as forward and back propagation.
# 
# parameters are typically:
#   size - integer of size of layer (number of activations)
#   act - activation function as defined by activations module
# so the initializer is actually an implementation for declaration of
# instance variables
# weights and biases not set by default
#
# all layers implement the LayerIterator class for forward
# propagation iteration
# 
# if you add a new class the implements the layer interface, be sure to add
# add it and a unique string identifier to the list of types
#


# system libraries
# internal libraries
from nucleos.iterators import LayerIterator
# third party libraries
import numpy as np


class Layer(object):

    
    def __init__(self, size, activ):
        self.size = size        # num nodes in layer
        self.activ = activ      # activation function
        
        self.w = None           # incoming weights  (size x prev-size) matrix
        self.b = None           # incoming biases   (size x 1) vector
        
        self.a = None           # incoming activation vector from previous layer
        self.z = None           # weighted sum + bias vector
        self.x = None           # activation function on z vector

        self.next_ = None       # layer that comes after this
        self.prev = None        # layer that comes before this
        
        self.delta_w = None   # partial weights set to none as no update required
        self.delta_b = None   # partial biases set to none for same reason

        self.type_ = type_gen
    

    # propagate forward while recording vectors for optimization
    # given activations of previous layer
    def forward(self, incoming_activations):    
        raise NotImplementedError

    # propagate backward while using vectors from forward
    # propagation given mu vector to be passed to next layer
    def _backward(self, mu):
        raise NotImplementedError
    
    # use only this method to add a layer to the list
    # append function should be the default interface
    def append(self, next_):
        raise NotImplementedError

    # Do not call this method from a non-layer caller.
    # prepend function that should also establish
    # the weight connection between the adjacent layers.
    def _prepend(self, prev):
        raise NotImplementedError

    # handle randomization of weights and biases
    # based on the layer's situation (ie preceding layer)
    # and type of layer
    def randomize_parameters(self):
        raise NotImplementedError

    # update the weights and biases for the layer
    # using the calculated cost gradient components
    # this function may not be used by all layers.
    # the delta matrices should be the same shape
    # as the weight and bias matrices.
    # after update, deltas should be zeroed.
    def _apply_updates(self, alpha):
        if self.b is not None and self.w is not None:
            self.b -= alpha * self.delta_b
            self.w -= alpha * self.delta_w

    # zero the delta matrices by setting them equal to
    # matrices of the same shape filled with zeros.
    # if they are none, just leave them as None
    def _zero_deltas(self):
        if self.w is not None and self.b is not None:
            self.delta_w = np.zeros(self.w.shape)
            self.delta_b = np.zeros(self.b.shape)
    
    # iterable to iterate forwards on across layers
    # returns specific LayerIterator class
    def __iter__(self):
        return LayerIterator(self)

    # returns a random n-dimensional array of 
    # of random values using numpy where n
    # is the number of dimensions. Each argument
    # is the size of the respective dimension
    def rand(self, *args):
        return np.random.randn(*args)

# types of layers to specify layer type property
# available in layers module
type_gen = "gen"
type_input = "input"
type_mlp = "mlp"



