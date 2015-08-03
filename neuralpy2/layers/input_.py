# 
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# Implementation of layer interface for
# input layer. retains super intitializer.
# no weight or bias caculations
# no backward propagation
# only append, no prepend (assuming this is the first layer)
# 

# system libraries
# internal libraries
import layer
# third party libraries
import numpy as np


class Input(layer.Layer):
    
    def __init__(self, size):
        super(Input, self).__init__(size, None)
        self.type_ = layer.type_input

    # Implementation of layer interface
    # for input vector layer, no calculations
    def forward(self, x):
        self.x = x
        return x

    # do nothing becaus ethere is no backward
    # propagation calculation for input layer
    # return None to indicate end of propagation
    def _backward(self, *args):
        return None

    # assign to instance var next_
    # prepend self to next_
    # caller beware: may potentially overwrite next_'s weights
    def append(self, next_):
        self.next_ = next_
        next_._prepend(self)


    # there are no incoming parameters for the 
    # input layer
    def randomize_parameters(self):
        self.w = None
        self.b = None