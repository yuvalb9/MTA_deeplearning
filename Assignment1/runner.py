import numpy as np


def mean_squared_error(actual, target):
    if len(actual) != len(target):
        raise ValueError('Wrong number of targets')

    return np.mean(np.power(np.subtract(actual, target), 2))


def identity(z, prime=False):
    if prime:
        return 1
    return z


def sigmoid(z, prime=False):
    if prime:
        f = sigmoid(z, prime=False)
        return np.multiply(f, (1.0 - f))
    return 1.0 / (1.0 + np.exp(-1.0 * z))


class Layer:
    def __init__(self, layer_name, input_size, layer_size, activation_function, weight_initializer=np.random.random):
        self.weight_initializer = weight_initializer
        self.input_size = input_size + 1  # added 1 for bias
        self.layer_size = layer_size
        self.activation_function = activation_function
        self.layer_name = layer_name
        self.weights = weight_initializer((self.input_size, self.layer_size))

    def feedforward(self, values):
        values_rows = values.shape[0]
        bias_and_values = np.concatenate([np.ones((values_rows, 1)), values], axis=1)
        pre_activation = np.dot(bias_and_values, self.weights)
        post_activation = self.activation_function(pre_activation)
        return values, pre_activation, post_activation


class FCNN():
    """fully connected neural network"""
    _layers = []

    def __init__(self, layers_list, cost_func, learning_rate):
        self.cost_func = cost_func
        self._layers = layers_list
        self.learning_rate = learning_rate

    def feed_forward(self, values_matrix):
        layers_data = []
        tmp = values_matrix
        for layer in self._layers:
            (input_data, pre_activation, post_activation) = layer.feedforward(tmp)
            tmp = post_activation
            layers_data.append({"input":input_data, "pre activation":pre_activation, "post activation":post_activation})

        return layers_data

    def train(self, data_input, labels):
        layers_data = self.feed_forward(data_input)
        layers_deltas = self.calculate_deltas(data_input, labels, layers_data)
        weights_deltas = self.calculate_weights_deltas(layers_deltas, layers_data)

        # update the weights
        for i in xrange(len(weights_deltas)):
            weight_delta = weights_deltas[i]
            layer_weight = self._layers[i].weights
            layer_weight -= self.learning_rate * weight_delta


    def calculate_deltas(self, data_input, target_data, layers_data):
        # type: (np.matrix, np.matrix, dict) -> list
        """
        handles the back propagation step of the Neural Network algorithm

        :param data_input: a matrix with rows of input data
        :param target_data: a matrix with rows of output labels
        :param layers_data: a dictionary with results from the feed forward step
        :return: a list of deltas
        """
        num_of_layers = len(self._layers)
        delta = []
        for i in reversed(xrange(num_of_layers)):
            layer_output = layers_data[i]['post activation']
            layer_input = layers_data[i]['pre activation']

            if i == num_of_layers-1:
                # handle output layer

                output_delta = np.subtract(layer_output, target_data)
                layer_delta = np.multiply(output_delta, self._layers[i].activation_function(layer_input, prime=True))
                error = np.sum(np.power(output_delta,2))
                print error
            else:
                # handle hidden layers
                next_layers_weights = self._layers[i+1].weights
                delta_pullback = next_layers_weights.dot(delta[-1].T)
                layer_delta = np.multiply(delta_pullback[1:,:].T, self._layers[i].activation_function(layer_input, prime=True))
            delta.append(layer_delta)


        return delta

    def calculate_weights_deltas(self, layers_deltas, layers_data):
        num_of_layers = len(self._layers)
        weights_deltas = []
        for index in range(num_of_layers):
            delta_index = num_of_layers - 1 - index
            values_rows = layers_data[index]['input'].shape[0]
            bias_and_values = np.concatenate([np.ones((values_rows, 1)), layers_data[index]['input']], axis=1)
            layer_weights_delta = bias_and_values.T.dot( layers_deltas[delta_index] )
            weights_deltas.append(layer_weights_delta)

        return weights_deltas

def targetFunction(x, y):
    return 3.0 * x + 4.5 * y


def generate_toy_sample(amount):
    x = [x * 20.0 - 10 for x in np.random.rand(amount)]
    y = [y * 20.0 + 10 for y in np.random.rand(amount)]
    labels = []
    for i in range(amount):
        t = targetFunction(x[i], y[i])
        labels.append(t)

    return x, y, np.array(labels)


if __name__ == '__main__':
    learning_rate = 0.1
    np.random.seed(101)  # will make sure we use the same randomized numbers
    inL = Layer("1st hidden", 2, 4, sigmoid)
    hidL = Layer("2st hidden", 4, 3, sigmoid)
    outL = Layer("output", 3, 1, identity)

    nn = FCNN([inL, hidL, outL], mean_squared_error, learning_rate)

    for i in xrange(10):
        x, y, tgt = generate_toy_sample(1)

        # input data is a matrix, with every row being an example to work on
        # labels is a matrix with every row being label for the corresponding  data_input
        input_data = np.matrix([x, y]).T
        labels = np.matrix(tgt).T

        #print input_data[0,:]
        #print labels[0,:]

        # print ("weight 1st hidden\n: ")

        nn.train(input_data, labels)
