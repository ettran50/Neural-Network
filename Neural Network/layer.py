# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for an input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes gradients
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError