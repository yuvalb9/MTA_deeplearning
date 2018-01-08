import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-1.0 * z))


class Layer():
    def __init__(self, layer_name, input_size, layer_size, activation_function, weight_initializer=np.zeros, biases_initializer=np.zeros):
        self.biases_initializer = biases_initializer
        self.weight_initializer = weight_initializer
        self.input_size = input_size
        self.layer_size = layer_size
        self.activation_function = activation_function
        self.layer_name = layer_name
        self.weights = weight_initializer( (layer_size, input_size) )
        self.biases  = biases_initializer( layer_size )

    def feed(self, values):
        print "Layer %s:"%(self.layer_name)
        print "\tweight:", self.weights
        print "\tbiases:", self.biases
        print "\tvalues:", values
        pre_activation = np.add( np.dot( self.weights, values) , self.biases )
        print "\tpre activation:", pre_activation
        post_activation = self.activation_function(pre_activation)
        print "\tpost activation:", post_activation
        return post_activation


class FCNN():
    '''fully connected neural network'''
    _layers = []

    def __init__(self, layers_list):
        self._layers = layers_list

    def feed_forward(self, values_matrix):
        ret = None
        tmp = values_matrix
        for layer in self._layers:
            tmp = layer.feed(tmp)
        ret = tmp
        return ret


if __name__ == '__main__':
    inL = Layer("input", 3, 4, sigmoid, weight_initializer=np.random.random)
    hidL = Layer("1st hidden", 4, 2, sigmoid)
    outL = Layer("output", 2,1, sigmoid)

    nn = FCNN([inL, hidL, outL])
    print nn.feed_forward(np.array([1,2,3]))


