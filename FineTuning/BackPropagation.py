import time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from TransferLearn.randomInitilize import initialize_network
from TransferLearn.initilizeModel import init


start = time.time()


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + np.exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    # print(neuron)
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            # neuron = layer
            # errors.append(expected - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch):
    error = []
    epoch_no = []
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            # print(row)
            outputs = forward_propagate(network, row)
            expected = row[-1]
            sum_error += 1/2 * (expected - outputs[0]) ** 2
            # sum_error += (outputs[0]*(1-outputs[0])**2)
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        error.append(sum_error)
        epoch_no.append(epoch)
        print('\t\t>epoch={}, learning rate={}, error={}'.format(epoch, l_rate, sum_error))
    # graphPlot(error, epoch_no)
    print('\t\t----*** Fine Tuning Model Trained ***----')


def graphPlot(loss, epoch):
    plt.title('Classifier status')
    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.xlim(0, 200)
    plt.ylim(0, 20)
    plt.plot(epoch, loss, color='red')
    plt.show()


model = init()
dataset = pd.read_csv('/home/prajwal/Projects/CancerDetector/normalizedData/TrainData.csv')
dataset = np.array(dataset)
train_network(network=model, train=dataset, l_rate=0.1, n_epoch=200)
print('\t\tTraining Time', (time.time() - start), 'seconds')
# print(model)
filename = '/home/prajwal/Projects/CancerDetector/classifier/experiments/fineTunedModel_06.pkl'
# pickle.dump(model, open(filename, 'wb'))
