import neuralpy2

training_set = [
    ([[1],[1]], [[1]]),
    ([[1],[0]], [[0]]),
    ([[0],[1]], [[0]]),
    ([[0],[0]], [[0]])
]

# alternatively you could write training_set = neuralpy2.load_and()

net = neuralpy2.Network([2, 3, 1])

for x, y in training_set:
	neuralpy2.output(net.forward(x))


epochs = 300
learning_rate = 1.0

net.train(training_set[:], epochs, learning_rate, mini_batch_size=2, monitor=True)

print "\n"

for x, y in training_set:
	neuralpy2.output(net.forward(x))

net.show_cost()
