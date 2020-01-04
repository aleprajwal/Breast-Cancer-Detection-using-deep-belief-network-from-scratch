import pickle
import json
import numpy as np


filename = './classifier/fineTunedModel.pkl'
network = pickle.load(open(filename, 'rb'))


def normalizeDataset(dataset):
    dataset = np.array(dataset)
    normalize = (dataset - 1) / (10 - 1)
    return normalize


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


def readJson(filename='./json/data.json'):
    dataset = []
    with open(filename, 'r') as file:
        data = json.load(file)
    for key, value in data.items():
        value = int(value)
        dataset.append(value)
    return dataset


if __name__ == '__main__':
    dataset = readJson()
    dataset = normalizeDataset(dataset)
    output = forward_propagate(network, dataset)
    result = {'result': output[0]}
    json.dump(result, open('./json/result.json', 'w'))
    print(output)
    if output[0] >= 0.7:
        print('Malignant')
    elif output[0] <= 0.3:
        print('Benign')
    else:
        print('Inconclusive')
