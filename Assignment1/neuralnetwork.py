import numpy as np

# transfer function

EPSILON = 1e-10




def sigmoid(x, derivative=False):
    x = np.clip(x, -500.0, 500.0)
    if not derivative:
        return 1 / (1 + np.exp(-x))
    else:
        out = sigmoid(x)
        return out * (1 - out)


def relu(x, derivative=False):
    if not derivative:
        return x * (x > 0)
    else:
        return 1.0 * (x > 0)


def softmax(x, derivative=False):
    if not derivative:
        return np.exp(x, axis=1) / (np.sum(np.exp(x), axis=1) )
    else:
        return softmax(x) * (1 - softmax(x))


def linear(x, derivative=False):
    if not derivative:
        return x
    else:
        return 1.0


def gaussian(x, derivative=False):
    if not derivative:
        return np.exp(-x ** 2)
    else:
        return -2 * x * np.exp(-x ** 2)


def tanh(x, derivative=False):
    if not derivative:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2


class BackPropagationNetwork:
    """A back propagation network"""

    layerCount = 0
    shape = None
    weights = []
    tFuncs = []

    def __init__(self, layer_size, layer_functions=None):
        """initialize the network"""

        # layer info
        self.layerCount = len(layer_size) - 1  # the input layer is not a real layer
        self.shape = layer_size

        if layer_functions is None:
            lfuncs = []
            for i in range(self.layerCount):
                if i == self.layerCount-1:
                    lfuncs.append(linear)
                else:
                    lfuncs.append(sigmoid)
        else:
            if len(layer_size)!= len(layer_functions):
                raise ValueError("Incompatible list of transfer functions")
            elif layer_functions[0] is not None:
                raise ValueError("input layer cannot have a transfer functions")
            else:
                lfuncs = layer_functions[1:]
        self.tFuncs = lfuncs



        # Input/Output data from last run
        self._layerInput = []
        self._layerOutput = []
        self._previouseWeightDelta = []

        # Create the weight arrays
        for (l1, l2) in zip(layer_size[:-1], layer_size[1:]):
            #self.weights.append(np.random.random(loc=0, scale=3.0, size=(l2, l1 + 1)))
            self.weights.append(2 * np.random.random(size=(l2, l1 + 1)) - 1)
            #self.weights.append(np.random.randn())
            self._previouseWeightDelta.append(np.zeros([l2, l1+1]))



    def run(self, input_matrix):
        """Run the network based on the input date"""
        # each row in input is an item
        ln_cases = input_matrix.shape[0]
        # clear out previous intermediate value lists
        self._layerInput = []
        self._layerOutput = []

        # run it
        for index in range(self.layerCount):
            if index == 0:
                layer_input = self.weights[0].dot(np.vstack([input_matrix.T, np.ones([1, ln_cases])]))
            else:
                layer_input = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, ln_cases])]))
            self._layerInput.append(layer_input)
            activation_function = self.tFuncs[index]
            self._layerOutput.append(activation_function(layer_input))

        return self._layerOutput[-1].T

    def train_epoch(self, input_matrix, target, training_rate=0.2, momentum=0.5):
        """this method trains the network for one epoch"""
        delta = []
        ln_cases = input_matrix.shape[0]

        #   first run the network
        self.run(input_matrix)

        # calculate little delta
        for index in reversed(range(self.layerCount)):
            activation_function = self.tFuncs[index]
            if index == self.layerCount - 1:
                # compare to the target values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * activation_function(self._layerInput[index], True))
            else:
                # compare to the following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * activation_function(self._layerInput[index], True))

        # compute weight deltas
        for index in range(self.layerCount):
            activation_function = self.tFuncs[index]
            delta_index = self.layerCount - 1 - index
            if index == 0:
                layer_output = np.vstack([input_matrix.T, np.ones([1, ln_cases])])
            else:
                layer_output = np.vstack(
                    [self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            curr_weight_delta = np.sum(
                layer_output[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0)
                , axis=0)

            weight_delta = training_rate * curr_weight_delta + momentum * self._previouseWeightDelta[index]
            self.weights[index] -= weight_delta
            self._previouseWeightDelta[index] = weight_delta

        return error


if __name__ == '__main__':

    lvInput = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    lvTarget = np.array([[0.0], [0.0], [1.0], [1.0]])
    tfuncs = [None, sigmoid, linear]

    bpn = BackPropagationNetwork((2, 2, 1), tfuncs)
    print(bpn.shape)
    print(bpn.weights)

    lvOutputBefore = bpn.run(lvInput)

    lnMax = 100000
    lnErr = 1e-8
    for i in range(lnMax - 1):
        err = bpn.train_epoch(lvInput, lvTarget)
        if i % 2500 == 0:
            print("iteration {0}\nError: {1:0.6f}".format(i, err))
        if err < lnErr:
            print("minimum error reached at iteration {0}".format(i))
            break

    lvOutputAfter = bpn.run(lvInput)
    print("input: {0}\nbefore:{1}\nAfter:{2}".format(lvInput, lvOutputBefore, lvOutputAfter))
