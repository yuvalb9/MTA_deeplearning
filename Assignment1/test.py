# Python Standard Library struct and array
# for dealing with reading dataset from file
import neuralnetwork

import struct
from array import array
from time import time

# Numpy for calculating
import numpy as np


# Some Useful Helper Functions

from pprint import pprint
""" helper functions """
def names_in(dictionary):
    """ list all names in a dictionary """
    print([name for name,_ in sorted(dictionary.items())])
def names_shape_in(dictionary):
    pprint([(name, val.shape) for name,val in sorted(dictionary.items())])

def debug_show_all_variables():
    global cache, parameters, hyper_parameters
    print("cache: ")
    names_shape_in(cache)
    print("parameters: ")
    names_shape_in(parameters)
    print("hyper_parameters: ")
    names_in(hyper_parameters)


def load_mnist():
    """
    load MNIST dataset into numpy array
    MNIST dataset can be downloaded manually.
    url: http://yann.lecun.com/exdb/mnist/
    """
    ret = {}
    with open('Data/train-images.idx3-ubyte', 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        assert(magic==2051)
        ret['X_train'] = np.array(array("B", f.read())).reshape(size,rows,cols)

    with open('Data/t10k-images.idx3-ubyte', 'rb') as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
        assert(magic==2051)
        ret['X_test'] = np.array(array("B", f.read())).reshape(size,rows,cols)

    with open('Data/train-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        assert(magic==2049)
        ret['Y_train'] = np.array(array("B", f.read())).reshape(size,1)

    with open('Data/t10k-labels.idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        assert(magic==2049)
        ret['Y_test'] = np.array(array("B", f.read())).reshape(size,1)

    return ret


def shuffle_divide_dataset(dataset, len_of_dev=10000):
    """
    Shuffle and divide the dataset

    len_of_dev: 10,000 is a reasonable number for dev set.
                Dev dataset with this size is big enough to measure variance problem.
    """
    assert ('X_train' in dataset)
    assert (len(dataset) == 4)

    """ random shuffle the training set """
    np.random.seed(1)
    permutation = np.random.permutation(dataset['X_train'].shape[0])
    dataset['X_train'] = dataset['X_train'][permutation]
    dataset['Y_train'] = dataset['Y_train'][permutation]

    """ divide trainset into trainset and devset """
    dataset['X_dev'] = dataset['X_train'][:len_of_dev]
    dataset['Y_dev'] = dataset['Y_train'][:len_of_dev]
    dataset['X_train'] = dataset['X_train'][len_of_dev:]
    dataset['Y_train'] = dataset['Y_train'][len_of_dev:]

    return dataset




def manually_validate_mnist_dataset(dataset):
    """Manually check the dataset by random visualize some of them"""
    random_train = np.random.randint(1, len(dataset['X_train']))-1
    random_dev = np.random.randint(1, len(dataset['X_dev']))-1
    random_test = np.random.randint(1, len(dataset['X_test']))-1
    print(dataset['Y_train'][random_train], dataset['Y_dev'][random_dev], dataset['Y_test'][random_test])
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize=[10,3])
    ax1.imshow(dataset['X_train'][random_train], cmap='gray')
    ax2.imshow(dataset['X_dev'][random_dev], cmap='gray')
    mappable = ax3.imshow(dataset['X_test'][random_test], cmap='gray')
    fig.colorbar(mappable)
    plt.show()




def standardize( dataset ):
    """use standard sccore to normalize input dataset"""
    assert('X_train' in dataset)
    mu = np.mean(dataset['X_train'], keepdims=False)
    sigma = np.std(dataset['X_train'], keepdims=False)
    for key, val in dataset.items():
        if key[:2]=='X_':
            dataset[key] = ( dataset[key] - mu ) / sigma
    return dataset



def flat_stack( dataset ):
    """input dataset format: (m, width, height)"""
    for key, val in dataset.items():
        if key[:2]=='X_':
            width = dataset[key].shape[1]
            height = dataset[key].shape[2]
            dataset[key] = dataset[key].reshape(-1, width*height).T
    return dataset



def one_hot( dataset ):
    min_label_number = np.min(dataset['Y_train'], keepdims=False)
    max_label_number = np.max(dataset['Y_train'], keepdims=False)
    C = max_label_number - min_label_number + 1
    for key, val in dataset.items():
        if key[:2]=='Y_':
            # all label number should be trained in Y_train
            assert(min_label_number <= np.min(dataset[key], keepdims=False))
            assert(max_label_number >= np.max(dataset[key], keepdims=False))
            Y = dataset[key]
            Y_onehot = np.zeros((C, Y.shape[0]))
            Y_onehot[Y.reshape(-1).astype(int), np.arange(Y.shape[0])] = 1
            dataset[key] = Y_onehot
    return dataset


def back_one_hot(Y_onehot):
    """ This is an inverse function of one hot, in case we need to interpret the result. """
    Y = np.repeat( [np.arange(Y_onehot.shape[0])], repeats=Y_onehot.shape[1], axis=0 )
    assert(Y.shape == Y_onehot.T.shape)
    Y = Y[Y_onehot.T.astype(bool)]
    return Y.reshape(-1,1)


def main():
    """test model using real MNIST dataset"""
    mnist = load_mnist()
    mnist = shuffle_divide_dataset(mnist)
    mnist = standardize(mnist)
    mnist = flat_stack(mnist)
    mnist = one_hot(mnist)

    bpn = neuralnetwork.BackPropagationNetwork((784, 30, 10), [None, neuralnetwork.sigmoid, neuralnetwork.sigmoid])


    lnMax = 100000
    lnErr = 1e-8
    input_data = mnist['X_train'].T
    labels = mnist['Y_train'].T
    cases = input_data.shape[0]
    batch_size = 10
    currpos = 0

    print(input_data.shape)
    run_validation(bpn, mnist)

    for i in range(lnMax - 1):
        # err = bpn.train_epoch(lvInput, lvTarget)
        batch_start = currpos
        batch_end = min(batch_start + batch_size, cases)
        batch_input = input_data[batch_start:batch_end,:]
        batch_labels = labels[batch_start:batch_end, :]


        err = bpn.train_epoch(batch_input, batch_labels, training_rate=0.1)
        if i % 2500 == 0:
            print("iteration {0}\nError: {1:0.6f}".format(i, err))
            run_validation(bpn, mnist)
        #if err < lnErr:
        #    print("minimum error reached at iteration {0}".format(i))
        #    break

        currpos += batch_size

def run_validation(net, mnist):
    records_cnt = mnist['Y_dev'].T.shape[0]
    output = net.run(mnist['X_dev'].T)
    target = mnist['Y_dev'].T
    max_x = np.argmax(output, axis=1)
    x_onehot = np.zeros((records_cnt, 10))
    x_onehot[np.arange(records_cnt ), max_x] = 1
    unique, counts = np.unique((target == x_onehot).all(axis=1), return_counts=True)
    results = dict(zip(unique, counts))
    print ("Precision: % {0:0.6f}".format(100.0000*(results[True])/(results[True]+results[False])))


if __name__ == '__main__':
    main()
