import math
import unittest

from NN.sigmoid_neuron import SigmoidNeuron


class TestNeuron(unittest.TestCase):

    def test_feed(self):
        input = [0.2, 0.2]
        bias = 0.5
        n = SigmoidNeuron(input, bias)
        weights = n.weights
        s = sum([i * w for i, w in zip(input, weights)])
        expected = 1 / (1 + math.exp(-(s + n.bias)))
        self.assertEqual(n.feed(input), expected)


