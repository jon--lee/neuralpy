.. |neuralpy| replace:: :mod:`neuralpy`
.. |true| replace:: :mod:`True`
.. |false| replace:: :mod:`False`
.. |0| replace:: :mod:`0`
.. |1| replace:: :mod:`1`

Welcome to neuralpy's documentation!
************************************
This page desribes how to get set up with |neuralpy| and how to start using it.
We always need help! Contribute on `Github <https://github.com/jon--lee/neuralpy.git>`_!


.. toctree::
   :maxdepth: 2



Getting Started (XOR example)
---------------
Let's start off with a more detailed and involved example than the quick start guide found in the `README <https://github.com/jon--lee/neuralpy/blob/master/README.rst>`_.

The goal for this example is to create a neural network that will replicate the archetypal `exclusive or <https://en.wikipedia.org/wiki/Exclusive_or>`_, XOR, function. This function is a "logical operation that outputs |true| only when both inputs differ (one is true, and the other is false)." You're probably familiar with this function from programming. Here's some psuedo-code for what it should theoretically accomplish::
	
	output = XOR(True, True)		// output --> False
	output = XOR(False, True)		// output --> True
	output = XOR(True, False)		// output --> True
	output = XOR(False, False)		// output --> False

For our example, we'll use :mod:`1` to represent |true| and :mod:`0` to represent |false|.

Getting started is pretty easy. Install |neuralpy| from PyPI by running the following command::

	$ pip install neuralpy

Then in your Python project, all you have to do is import it::

	import neuralpy

Now, let's create a network and start training it. First we'll need to pass a series of non-zero integers that determines how many nodes we're going to have in each layer. |neuralpy| allows us to have as many layers as we want, so let's use four for demonstration. As you get comfortable with neural networks, you'll start to figure out how many hidden layers you need and how many neurons per hidden layer you need for certain situations. As far as we know now, there's no real right or wrong answer, but some combinations work better than others. Remember: for XOR, we have to give the function two inputs and we expect one output::

	net = neuralpy.Network(2, 10, 8, 1)

.. note::

	The |neuralpy| class :mod:`Network` can take either a list of non-zero integers or just a series of non-zero integers separated by commas. So this code::

		layers = [2, 10, 8, 1]
		net = neuralpy.Network(layers)

	is identical to the code before it. It's up to you how you want to initialize the network.


That code creates a neural network with two input nodes, one hidden (or intermediate) layer with three neurons, and an output layer with one neuron. Technically, we call this a three layer network even though there are four actual layers because there are only three layers of processing units (the input layer does not process anything).

|neuralpy| will automatically generate random incoming weights and biases for each processing layer.

The function :mod:`feedforward` takes a list representing a vector of inputs and reutrns a list representing the output vector that the network calculates. Let's see what happens when we give our network some inputs::

	print net.feedforward([1,0])
	# ex output: [0.46902402362712664]

Okay, so you may have gotten something different from me. But, like me, you probably didn't get an output that said :mod:`[1.0]`, which is what we would expect from XOR when we give it |true| and |false|.

Try out some of the other inputs that we defined like |false| and |false|.

Still no luck?

Well, that's why we train neural networks! The purpose of neural networks is to give them a "training set" which is a series of inputs and their corresponding outputs. The network uses this information to adjust its weights and biases so that when you give it one of those inputs, it will produce the appropriate output. It "minimizes the cost function," which essentially means that it closes the difference between what it is outputting currently (with incorrect weights and biases) and what it should be outputting (with correct weights and biases). It's converging to its "minimum" in the cost function because we want our error to be as low as possible.

We have to show our network what XOR looks like so it can gradually learn how to produce that function. Let's create a list of training data::

	datum_1 = ([1, 1], [0])
	datum_2 = ([1, 0], [1])
	datum_3 = ([0, 1], [1])
	datum_4 = ([0, 0], [0])

	training_data = [datum_1, datum_2, datum_3, datum_4]

Our training data is just a list of tuples where, in each tuple, there is first a list of inputs that we want to give it and then the output that we expect from that input.

Now we need to come up with some of the other hyperparameters. First let's say that we only want the neural network to train for a certain number of iterations and no more. We call these iterations "epochs" and they're kind of synonymous with time, but since computation time is different for everyone, we can universally use epochs instead::

	epochs = 300

Now we want to set our learning rate which is a factor that basically scales the amount that we adjust each weight and bias during every iteration. Too high of a learning rate may overshoot our minimum. Too low of a learning rate may make our network's convergence too slow. You just have to play around with it to get it right, but for now we'll say it's just :mod:`1`::

	learning_rate = 1

Now we have all of the basic requirements ready to start training the network. You could now just add::

	net.train(training_data, epochs, learning_rate, monitor_cost = True)

It may take a few seconds to train depending on your system and python implementation. Notice that I added :mod:`monitor_cost = True`. This is an optional parameter but you can use it to track the cost after every epoch.

You can then call the :mod:`show_costs()` function which will open a :mod:`matplotlib pyplot` showing you the progress of your network as it trains on the data you gave it::

	net.show_costs()

Now that we've trained the network and taken a look at the cost function, let's see what the network produces for the |true| and |false| input that we gave it earlier::

	print net.feedforward([1, 0])
	# ex: [0.946417057338419]

Well, it's not exactly |1| but it's pretty darn close! That's the thing with neural networks. They're approximators. You'll rarely get an integer as a result, but the point is you can round to the nearest integer or do some other post-processing.

Let's see what all the inputs in our :mod:`training_data` produce::

	for datum in training_data:
		x = datum[1]
		print net.feedforward(x)
		# ex: 	
		#	[0.06214085086576566]
		#	[0.946417057338419]
		#	[0.9352235744480635]
		#	[0.05643177490633071]

Not bad!

Neural Networks have a lot of applications. Once you have a model like |neuralpy| it's just about feature selection and pre-processing now. In fact, with this architecture, you can start doing optical character recognition easily!


- *"Good news! Your biscuits have arrived! They've been approved from the Wellington office."*
- *"I got a rejected form."*
- *"Aw, Jemaine. Rejected. Let's have a look. 'Did not fill out the form correctly: Purpose for the biscuits.' And you've put NA? What is NA?"*
- *"Not applicable."*
- *"What? There's no purpose for your biscuits?"*
- *"No, I just wanted them."*
- *"Well, they're hardly gonna send ya biscuits if there's no purpose! Think about it. Fill out your forms properly."*
- *"Well, I probably would have eaten them, I suppose."*
- *"Bret, what did you put on your form?"*
- *"I think I put I was gonna eat them."*








