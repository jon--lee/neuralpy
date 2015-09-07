import neuralpy

net = neuralpy.Network([2, 10, 8, 1])

print net.forward([1, 0])

datum_1 = ([1, 1], [0])
datum_2 = ([1, 0], [1])
datum_3 = ([0, 1], [1])
datum_4 = ([0, 0], [0])

training_set = [datum_1, datum_2, datum_3, datum_4]

# alternatively you could write training_set = load_xor()

epochs = 300
learning_rate = 1
net.train(training_set, epochs, learning_rate, monitor=True)


for x, y in training_set:
    print net.forward(x)
    # ex: 	
    #	[0.06214085086576566]
    #	[0.946417057338419]
    #	[0.9352235744480635]
    #	[0.05643177490633071]
net.show_cost()
