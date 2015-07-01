.. |neuralpy| replace:: :mod:`neuralpy`

Welcome to neuralpy's documentation!
************************************
This page desribes how to get set up with |neuralpy| and how to start using it.
We always need help! Contribute on `Github 
<https://github.com/jon--lee/neuralpy.git>`_!


.. toctree::
   :maxdepth: 2



Getting Started
---------------
Getting started is pretty easy! Install |neuralpy| from PyPI by running the following command::

	$ pip install neuralpy

Then in your Python project, all you have to do is import it::

	import neuralpy

Now, let's create a network and start training it. First we'll need to create a list of integers that determines how many nodes we're going to have in each layer. We can have as many layers as we want, but let's stick with three for now::

	layers = [2, 3, 1]
	net = neuralpy.Network(layers)

That code creates a neural network with two input nodes, one hidden (or intermediate) layer with three neurons, and an output layer with one neuron. Technically, we call this a two layer network even though there are three actual layers because there are only two layers of processing units (the input layer does not process anything).

|neuralpy| will automatically generate random incoming weights and biases for each processing layer.

More coming soon, sorry...

It's business time. *"You know when I'm down to just my socks it's time for business that's why they call it business socks."*
