# 
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
#
# iterator interface to be implemented by iterator
# classes such as LayerIterator. Do no implement
# these functions.
# makes implemented classes iterable.
#

# system libraries
# internal libraries
# third party libraries


class Iterator(object):
    def __init__(self, *args):
        raise NotImplementedError
    def __iter__(self):
        raise NotImplementedError
    def next(self):
        raise NotImplementedError
