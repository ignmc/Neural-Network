from argparse import ArgumentError


class NeuralNetwork:
    def __init__(self, first_layer=None, layers=None):
        if layers is None:
            if first_layer is None:
                raise ArgumentError("No layers supplied")

            # Case: only the first layer of an already built network was supplied
            self.first_layer = first_layer
            temp_layer = first_layer
            while temp_layer.next_layer is not None:  # Find the output layer
                temp_layer = temp_layer.next_layer
            self.last_layer = temp_layer
        else:
            # All the layers were supplied. Let's connect them in the same order they are listed
            for i, in range(len(layers) - 1):
                layers[i].next_layer = layers[i+1]
                layers[i+1].previous_layer = layers[i]
            self.first_layer = layers[0]
            self.output_layer = layers[-1]

    def feed(self, inputs):
        return self.first_layer.feed(inputs)

    def backward_propagate_error(self, expected_outputs):
        self.last_layer.backward_propagate_error(expected_outputs)

    def update_weights(self, inputs, leraning_rate):
        self.first_layer.update_weights(inputs)

    def train(self, inputs, expected_outputs, learning_rate):
        outputs = self.feed(inputs)
        self.backwards_propagate_error(expected_outputs)
        self.update_weights(inputs, learning_rate)
