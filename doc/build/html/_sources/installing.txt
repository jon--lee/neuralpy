.. |neuralpy| replace:: :mod:`neuralpy`
.. |neuralpy2| replace:: :mod:`neuralpy2`
.. |true| replace:: :mod:`True`
.. |false| replace:: :mod:`False`
.. |0| replace:: :mod:`0`
.. |1| replace:: :mod:`1`
.. |pip| replace:: :mod:`pip`
.. |matplotlib| replace:: :mod:`matplotlib`
.. |numpy| replace:: :mod:`numpy`
.. |easy_install| replace:: :mod:`easy_install`
.. |cd| replace:: :mod:`cd`

Installing
----------

It's recommended that you use |pip| to install |neuralpy|. There are also alternative methods
for doing so listed below.

.. note::
    You'll need |numpy| and |matplotlib| in order to run |neuralpy|. Those will be downloaded automatically
    if you install from |pip|, which is why it's recommended. At this time, it is a known issue that |numpy| and |matplotlib| do not automatically install
    as dependencies when using alternative methods of installation.

    If you would like to use easy_install you'll have to manually install |numpy|. If you are downloading the compressed archive,
    you must install |numpy| and |matplotlib|
    manually.


Install with |pip|::

    $ pip install neuralpy

or download the `compressed archive <https://pypi.python.org/pypi/neuralpy/>`_ from PyPI, expand it and |cd| into the directory that
contains **setup.py** and execute (be sure to install |numpy| and |matplotlib| manually as well)::

    $ python setup.py install
