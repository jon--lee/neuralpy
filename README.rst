neuralpy
--------


**neuralpy** is a neural network model written in python based on Michael Nielsen's neural networks and deep learning book.
This package provides a simple yet powerful fully-connected multilayer neural network. Since, this is a multilayer feedforward neural network, it is a universal approximator (Hornik, Stinchcombe and White, 1989). Neural Networks can be used for a wide range of applications from image processing to time series prediction.

Visit the (unfinished) `documentation page
<http://pythonhosted.org/neuralpy/>`_ or get started with the quick start guide below.

Getting Started (quick start)
+++++++++++++++++++++++++++++
Download and install **neuralpy** by running the following command::

	$ pip install neuralpy

Then in your python project you an import it and create network by passing it a list of integers that represent the number of nodes in each layer. You can have as many hidden (intermediate) layers as you want but this example will use just one::
	
	import neuralpy
	layers = [2, 3, 1]
	net = neuralpy.Network(layers)

The first integer in the list represents the length of the input layer and the last integer represents the length of the output layer. Integers between the two represent lengths of hidden layers going from left to right.

You can get the output of the network given a column vector::

	x = np.array([[1], [1]])
	output = net.feedforward(x)

``output`` will be a column vector.

More to come about training the network...

