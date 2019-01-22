import numpy as np
from activations import activations


class Layer:
    """ Class which models a neural network's generic layer.

        Attributes:

            input(np.array) - Input to the layer
            net(np.array) - Dot product between input and layer's weights
            out(np.array) - the output of the layer(= f(net))

            activation(str) - name of the activation function
            f(func) - the activation function to use
            d_f(func) - the derivative of the activation function (if exists)

            num_units(int) - number of units of the layer
            input_size(int) - shape of the input

            weights(np.array) - matrix of the weights of the layer (initialized randomly)
            bias(np.array) - the bias of the layer

    """
    def __init__(self, num_units, activation, input_size):

        self.input = None
        self.net = None
        self.out = None

        self.activation = activation
        self.f = activations(activation)[0]
        self.d_f = activations(activation)[1]

        self.num_units = num_units
        self.input_size = input_size

        self.weights = None
        self.bias = np.zeros((1, num_units))

    def init_weights(self, fan_out):
        """
        Initializes the weights by Glorot uniform.
        :param fan_out: number of neurons of the next layer
        """
        if self.activation == 'sigmoid' or self.activation == 'tanh':
            fan_in = self.input_size
            if fan_out is None:
                fan_out = self.num_units

            r = np.sqrt(6/(fan_in+fan_out))
            if self.activation == 'sigmoid':
                r = 4*r
            self.weights = np.random.uniform(low=-r, high=r, size=(self.input_size, self.num_units))

