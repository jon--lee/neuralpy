# system libraries
# internal libraries
# third party libraries
import numpy as np


# Interface for activation function. Should implement
# func which is the regular function, deriv which is the
# derivative of the function, and optional fast_deriv which
# is an optimized form of div that takes a different param
class Activ(object):
    def func(self,x):
        raise NotImplementedError
    def deriv(self,x):
        raise NotImplementedError
    def fast_deriv(self,y):
        raise NotImplementedError


# Implements activ for LINEAR function
# no fast derivative
class Lin(Activ):
    def func(self, x):
        return x
    
    def deriv(self, x):
        return 1.0

    
# Implements activ for SIGMOID (logistic) function
# includes fast derivative
class Sig():
    def func(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def deriv(self, x):
        y = self.func(x)
        return y * (1.0 - y)

    def fast_deriv(self, y):
        return y * (1.0 - y)


# EXPORTS of instances
linear = Lin()
sigmoid = Sig()

mapping = {
    "sigmoid": sigmoid,
    "linear": linear
}