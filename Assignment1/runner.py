import numpy as np

def sigmoid(z, prime=False):
    if prime:
        f = sigmoid(z, prime=False)
        return np.multiply(f , ( 1.0 - f ) )

    return 1.0/(1.0 + np.exp(-1.0 * z))


class Layer():
    def __init__(self, layer_name, input_size, layer_size, activation_function, weight_initializer=np.zeros):
        self.weight_initializer = weight_initializer
        self.input_size = input_size + 1 # added 1 for bias
        self.layer_size = layer_size
        self.activation_function = activation_function
        self.layer_name = layer_name
        self.weights = weight_initializer( ( self.input_size, self.layer_size) )


    def feedforward(self, values):
        print ("Layer %s:"%(self.layer_name))
        print ("\tweight:", self.weights)
        print ("\tvalues:", values)
        values_rows = values.shape[0]
        bias_and_values = np.concatenate(  [np.ones((values_rows, 1)) , values ], axis=1)
        pre_activation = np.dot( bias_and_values, self.weights)
        #print "\tpre activation:", pre_activation
        post_activation = self.activation_function(pre_activation)
        #print "\tpost activation:", post_activation
        return post_activation


def MSE(actual, target):
    if len(actual) != len(target):
        raise ValueError('Wrong number of targets')

    return np.mean(  np.power( np.subtract(actual, target), 2) )




class FCNN():
    '''fully connected neural network'''
    _layers = []

    def __init__(self, layers_list, cost_func):
        self.cost_func = cost_func
        self._layers = layers_list



    def feed_forward(self, values_matrix):
        layers_output = []
        tmp = values_matrix
        for layer in self._layers:
            tmp = layer.feedforward(tmp)
            layers_output.append( tmp )

        return layers_output


    def train_single(self, x, target, learning_rate = 0.1):
        layers_output = self.feed_forward(x)
        out = layers_output[-1]
        cost = self.cost_func(out, target)

        #print ("x:", x)
        #print ("out:", out)
        #print ("target:", target)
        #print ("cost:", cost)

        layers_deltas = self.backpropogate(layers_output, target)
        print ("layers_deltas:",layers_deltas )
        print ("layers_output:", layers_output)

        for i in range(len(layers_deltas)):
            if i==0:
                pass #learning_rate *

        return cost

    def backpropogate(self, layers_out, target):
        '''
        Calculates the Deltas for all layers
        :param layers_out: the output of each layer, after the activation function
        :param target: the label for the specific example, that generated the outputs
        :return: a list of delta values for every layer.
        '''
        layers_deltas = []
        tgt = np.asmatrix(target)   # change Vector to Matrix

        # get the Delta of the output layer
        out_delta =  tgt - layers_out[-1]
        layers_deltas.append( out_delta )

        print ("target: ", np.asmatrix(target))
        print ("layers_out[-1]: ", layers_out[-1])
        print ("out_delta:",out_delta)
        print("len(layers_out):",len(layers_out))

        # now go over every hidden layer, and calculate Deltas for their neurons
        for i in range(len(layers_out)-1,0-1,-1):
            curr_weights = self._layers[i].weights[1:,:] # the first delta is for the Bias, we dont push that down to
                                                         # lower layers


            delta = np.dot( curr_weights, layers_deltas[-1].T).T
            layers_deltas.append(delta)

        layers_deltas.reverse()     # reverse order, because we worked form end to start in the BP stage
        layers_deltas.pop(0)        # there is no need for Delta for input layer.
        return layers_deltas

def targetFunction(x,y,z):
    return 3.0*x + 4.5*y - 5.0*z


if __name__ == '__main__':
    inL = Layer("1st hidden", 3, 4, sigmoid, weight_initializer=np.random.random)
    hidL = Layer("2st hidden", 4, 2, sigmoid, weight_initializer=np.random.random)
    outL = Layer("output", 2, 1, sigmoid, weight_initializer=np.random.random)

    nn = FCNN([inL, hidL, outL], MSE)

    x = [x*20.0-10 for x in np.random.rand(100)]
    y = [y*20.0-10 for y in np.random.rand(100)]
    z = [z*20.0-10 for z in np.random.rand(100)]
    tgt = []
    for i in range(100):
        t =  np.array( [targetFunction(x[i], y[i], z[i]),] )
        tgt.append( t )

    t = tgt[0]
    i=0
    print ("train single:", nn.train_single(np.matrix([x[i],y[i],z[i]]), t))

