# system libraries
# internal libraries
# third party libraries
import numpy as np

# Cost interface that abstracts both the normal,
# generic function and derivative of that function
# deriv is assumed to ALWAYS be the analytical derivative
class Cost():
    def func(self, *args):
        raise NotImplementedError
    def deriv(self, *args):
        raise NotImplementedError

# implementation of cost interface for mean
# squared error which is defined as the one half the square
# of the difference between the actual and expected
class MeanSquare(Cost):
    
    # normal function defined as one
    # half of the mean square error
    def func(self, x, y):
        return 0.5 * ((x - y) ** 2)
    
    # derivative of the normal function
    def deriv(self, x, y):
        return x - y

mean_square = MeanSquare()

types = {
    "mean_square": MeanSquare()
}