from NN.neural_network import NeuralNetwork
import pandas as pd
import matplotlib.pyplot as plt

DATA = {
    # input: expected_output
    (1, 1): (0, 1),
    (1, 0): (1, 0),
    (0, 1): (1, 0),
    (0, 0): (0, 0)
}


def test_network(network, data_set):
    hits = 0
    for d in data_set:
        output = network.feed(d)
        result = evaluate_output(output, data_set[d])
        if result:
            hits += 1
    return hits/len(data_set)


def experiment(epochs):
    network = NeuralNetwork(layers_sizes=(2, 3, 4, 2))  # 2 hidden layers, 2 output values

    results = []

    for n_epoch in range(epochs):
        for data in DATA:
            network.train(data, DATA[data], 0.5)
        precision = test_network(network, DATA)
        results.append(precision)
    df = pd.DataFrame({'epochs': list(range(epochs)),
                       'precision': results})
    plt.figure()
    df.plot(x='epochs', y='precision')
    plt.show()


def evaluate_output(output, expected):
    return (output[0] > output[1]) == (expected[0] > expected[1])
