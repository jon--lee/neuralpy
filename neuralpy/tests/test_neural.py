import unittest
import neuralpy


class associatingElementsTest(unittest.TestCase):

	def test_cost_function(self):
		#tuples are arranged as ((input1, input2), expected_output)
		cases = [((1.0, 1.0), 0.0),
				((1.0, 0.5), 0.125),
				((0.5, 1.0), 0.125),
				((0.25, 0.0), 0.03125),
				((0.0, 0.25), 0.03125)
			]	
		for case in cases:
			input1 = case[0][0]
			input2 = case[0][1]
			output = case[1]
			self.assertEqual(neuralpy.cost_function(input1, input2), output)


if __name__ == '__main__':
	unittest.main()