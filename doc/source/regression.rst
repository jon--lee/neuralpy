.. |neuralpy| replace:: :mod:`neuralpy`
.. |true| replace:: :mod:`True`
.. |false| replace:: :mod:`False`
.. |0| replace:: :mod:`0`
.. |1| replace:: :mod:`1`


Regression tutorial
------------

Another simple application of |neuralpy| is regression. The key characteristic of neural
networks is that they are universal approximators.

So let's build a network that models
a simple sin function, which takes one input and has one output. Ideally we should get a network that
look something like this::

    >>> sin(0)
    0.0
    >>> sin(PI/2)
    1.0

In this tutorial we will:

    * generate our own inputs and outputs
    * normalize our training data
    * create a layer with a linear activation function
    * append that layer to our network
    * monitor the loss minimization
    * plot the final function


Importing the tools
~~~~~
We'll need neuralpy (obviously), matplotlib to plot things and numpy to generate our data::

    import matplotlib
    import numpy as np
    import neuralpy

Generating and normalizing training data
~~~~~
