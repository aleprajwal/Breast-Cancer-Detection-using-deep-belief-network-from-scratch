import numpy as np
from BasicTool.LoadDataset import loadDataset
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle


# filename = '/home/prajwal/Projects/CancerDetector/classifier/experiments/preTrainedModel_04.pkl'
filename = '/home/prajwal/Projects/CancerDetector/classifier/experiments/fineTunedModel.pkl'
network = pickle.load(open(filename, 'rb'))


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


if __name__ == '__main__':
    dataset, target = loadDataset('/home/prajwal/Projects/CancerDetector/normalizedData/TestData.csv')
    dataset = np.array(dataset)
    target = np.array(target)
    output = []
    for row in dataset:
        output.append(forward_propagate(network, row))
    for row in output:
        if row[0] > 0.5:
            row[0] = 1
        else:
            row[0] = 0
    output = np.array(output)
    confusion = confusion_matrix(target, output, labels=[1, 0])
    # print(confusion)
    True_Negative = confusion[1, 1]
    False_Positive = confusion[1, 0]
    False_Negative = confusion[0, 1]
    True_Positive = confusion[0, 0]
    print(pd.DataFrame(confusion, index=['Actual:Malignant', 'Actual:Benign'],
                       columns=['Predicted:Malignant', 'Predicted:Benign']))

    accuracy = (True_Positive + True_Negative)/(True_Positive + False_Positive + False_Negative + True_Negative)
    sensitivity = True_Positive/(True_Positive + False_Negative)
    specificity = True_Negative/(True_Negative + False_Positive)
    f1_measure = (2*specificity * sensitivity)/(specificity + sensitivity)
    print("\n\nClassifier Accuracy : {}%\n".format(accuracy * 100),
          "Sensitivity : {}%\n".format(sensitivity * 100),
          "Specificity : {}%\n".format(specificity * 100),
          "F1-Measure : {}%\n".format(f1_measure * 100))

