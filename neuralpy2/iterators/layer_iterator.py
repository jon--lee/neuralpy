#
# @author jonathan_lee@berkeley.edu (Jonathan N. Lee)
# 
# module that implementes the iterator interface designed
# specifically to iterate over layers which are similar to
# linked lists.
#
# for layer to implement this class add:
# 
#   def __iter__(self):
#       return LayerIterator(<iterable>)
#


# system libraries
# internal libraries
from iterator import Iterator
# third party libraries

class LayerIterator(Iterator):

    # initializer a layer iterator just like
    # a linked list where the iterator has a root
    # object
    def __init__(self, root):
        self.root = root

    def __iter__(self):
        return self

    def _len__(self):
        length = 0
        it = iter(self)
        for next_ in it:
            length += 1
        return length

    # return the next forward propagation of the layer
    # unless the layer is none then stop iteration.
    def next(self):
        if self.root is None:
            raise StopIteration
        else:
            temp = self.root
            self.root = self.root.next_
            return temp
