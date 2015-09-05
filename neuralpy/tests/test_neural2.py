import unittest
import neuralpy2

class associatingElementsTest(unittest.TestCase):

    # --TODO--
    # please update this function to adhere to the unittest rules
    # for now this will only fail when an exception is raised, not if
    # the code is bad.
    def test_forward(self):
        net = neuralpy2.Network([2, 3, 1])
        cases = [
                    ( [1, 1], None ),
                    ( [0, 0], None ),
                    ( [-1, 0], None ),
                    ( [-1, 1], None ),
                    #( [1, .75, .5], net.start.next_)
                ]
        for case in cases:
            inputs_ = case[0]
            if case[1] is not None:
                net.forward(inputs_, layer=case[1])
            else:
                net.forward(inputs_)


if __name__ == '__main__':
	unittest.main()
