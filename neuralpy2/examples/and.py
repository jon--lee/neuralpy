import nucleos

training_set = [
    ([[1],[1]], [[1]]),
    ([[1],[0]], [[0]]),
    ([[0],[1]], [[0]]),
    ([[0],[0]], [[0]])
]

net = nucleos.Network([2, 3, 1])

for x, y in training_set:
	nucleos.output(net.forward(x))


epochs = 300
learning_rate = 1.0

net.train(training_set[:], epochs, learning_rate)

print "\n"

for x, y in training_set:
	nucleos.output(net.forward(x))