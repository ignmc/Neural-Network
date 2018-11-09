from NN.neural_network import NeuralNetwork
import pandas as pd
import matplotlib.pyplot as plt
from timeit import default_timer as timeit

DATA = {
    # input: expected_output
    (1, 1): (0, 1),
    (1, 0): (1, 0),
    (0, 1): (1, 0),
    (0, 0): (0, 0)
}


def test_network(network, rows, expected_output):
    hits = 0
    for d, eo in zip(rows, expected_output):
        output = network.feed(d)
        result = evaluate_output(output, eo)
        if result:
            hits += 1
    return hits / len(rows)


def experiment(dataset, labels, epochs, layers_sizes, lr):  # try with layers_sizes = (11, 4, 5, len(possible_outputs))
    network = NeuralNetwork(layers_sizes=layers_sizes)  # 2 hidden layers, 2 output values

    results = []
    train_data = dataset[:int(len(dataset) // 2)]
    train_labels = labels[:int(len(labels) // 2)]
    query_data = dataset[int(len(dataset) // 2):]
    query_labels = labels[int(len(labels) // 2):]

    print("Training with {} epochs".format(epochs))
    start = timeit()
    for n_epoch in range(epochs):
        print("epoch: {}".format(n_epoch))
        if n_epoch != 0:
            for data, expected in zip(train_data, train_labels):
                network.train(data, expected, lr)
        precision = test_network(network, query_data, query_labels)
        results.append(precision)
    end = timeit()
    print("Time for {} epochs: {}s".format(epochs, end-start))
    df = pd.DataFrame({'epochs': list(range(epochs)),
                       'precision': results})
    plt.figure()
    df.plot(x='epochs', y='precision')
    plt.show()


def evaluate_output(output, expected):
    return output.index(max(output)) == expected.index(max(expected))
