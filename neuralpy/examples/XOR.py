import neuralpy

net = neuralpy.Network(2, 10, 8, 1)

neuralpy.output(net.feedforward([1,1]))
neuralpy.output(net.feedforward([0,1]))
neuralpy.output(net.feedforward([1,0]))
neuralpy.output(net.feedforward([0,0]))


datum_1 = ([1, 1], [0])
datum_2 = ([1, 0], [1])
datum_3 = ([0, 1], [1])
datum_4 = ([0, 0], [0])

training_data = [datum_1, datum_2, datum_3, datum_4]

epochs = 300

learning_rate = 1

net.train(training_data, epochs, learning_rate, monitor_cost=True)

neuralpy.output()

neuralpy.output(net.feedforward([1,1]))
neuralpy.output(net.feedforward([0,1]))
neuralpy.output(net.feedforward([1,0]))
neuralpy.output(net.feedforward([0,0]))

net.show_costs()