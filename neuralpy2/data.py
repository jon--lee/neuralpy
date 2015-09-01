import numpy as np
def load_and():
    inputs = [ [1, 1], [1, 0], [0, 1], [0, 0]]
    outputs = [[1], [0], [0], [0]]
    return zip(inputs, outputs)

def load_or():
    inputs = [ [1, 1], [1, 0], [0, 1], [0, 0]]
    outputs = [[1], [1], [1], [0]]
    return zip(inputs, outputs)

def load_xor():
    inputs = [ [1, 1], [1, 0], [0, 1], [0, 0]]
    outputs = [[0], [1], [1], [0]]
    return zip(inputs, outputs)

