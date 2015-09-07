neuralpy 1.3.0
--------------

**Within this package is the most intuitive fully-connected multilayer neural network model. Data science shouldn't have a high barrier to entry. neuralpy handles the math and overhead while you focus on the data.**

neuralpy is a neural network model written in python based on Michael Nielsen's neural networks and deep learning book.

- Visit the `neuralpy website <http://jon--lee.github.io/neuralpy/>`_
- Get detailed examples and explanations in the `Official Documentation <http://pythonhosted.org/neuralpy/>`_
- Contribute on `Github <https://github.com/jon--lee/neuralpy>`_

Getting Started (quick start)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The following demonstrates how to download and install **neuralpy** and how to create and train a simple neural network.
Run the following command to download and install::

	$ pip install neuralpy

Create a neural network in your project by specifying the number of nodes in each layer. Random weights and biases will automatically be generated::
	
	import neuralpy
	net = neuralpy.Network([2, 3, 1])

The network feeds input vectors as python lists forward and returns the output vector as a list::

	x = [1, 1]
	output = net.forward(x)
	print output
        # ex: [0.11471727263613461]

Train the neural network by first generating training data in the form of a list of tuples. Each tuple has two components and each component is a list representing the input and output respectively. This training set represents the simple OR function
and it can be generated for you to save typing::

	training_data = neuralpy.load_or()
        # [
        #   ([1, 1], [1]),
        #   ([1, 0], [1]),
        #   ...
        # ]

Then we must specify the remaining hyperparameters. Let's say we want to limit it to 100 epochs and give it a learning rate of 1::

	epochs = 100
	learning_rate = 1

Then run the *train* method with the parameters. We're telling the network to conform to training data::

	net.train(training_data, epochs, learning_rate)

Now feed forward the input from earlier and the output should be closer to 1.0, which is what we trained the network to do::

	output = net.forward(x)
	print output		# ex: [0.9542129706170075]

There is more information about advanced options such as monitoring the cost in the official documentation.

Since, this is a multilayer feedforward neural network, it is a universal approximator (Hornik, Stinchcombe and White, 1989). Neural Networks can be used for a wide range of applications from image processing to time series prediction.

- *"You abandoned me. You left me to die."*
- *"Well, I wouldn't have done it if I'd known you were going to hassle me about it."*
