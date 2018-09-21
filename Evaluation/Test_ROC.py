import matplotlib.pyplot as plt
import numpy as np
from BasicTool.LoadDataset import loadDataset
from sklearn.metrics import confusion_matrix
import pandas as pd
import pickle


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
def forwardPropagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def ROC_Curve(tpr, fpr):
    plt.title('Reciever Operating Characteristic (ROC) Curve')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    x_axis = fpr
    y_axis = tpr
    plt.plot([0, 100], [0, 100], color='red', linestyle='dashed')
    # plt.scatter(x_axis, y_axis)
    plt.plot(x_axis, y_axis, color='Blue')
    plt.show()


def confusionMatrix(target, result):
    confusion = confusion_matrix(target, result, labels=[1, 0])
    print(pd.DataFrame(confusion, index=['Actual:Malignant', 'Actual:Benign'],
                       columns=['Predicted:Malignant', 'Predicted:Benign']))
    return confusion


def evaluateModel(confusion):
    True_Negative = confusion[1, 1]
    False_Positive = confusion[1, 0]
    False_Negative = confusion[0, 1]
    True_Positive = confusion[0, 0]

    accuracy = (True_Positive + True_Negative)/(True_Positive + False_Positive + False_Negative + True_Negative)
    sensitivity = True_Positive / (True_Positive + False_Negative)
    specificity = True_Negative / (True_Negative + False_Positive)
    False_Positive_Rate = 1 - specificity
    f1_measure = (2*specificity * sensitivity)/(specificity + sensitivity)
    print("\n\nClassifier Accuracy : {}%\n".format(accuracy * 100),
              "Sensitivity : {}%\n".format(sensitivity * 100),
              "Specificity : {}%\n".format(specificity * 100),
              "1-Specificity : {}%\n".format(100-specificity * 100))
    return sensitivity*100, False_Positive_Rate*100


if __name__ == '__main__':
    dataset, target = loadDataset('/home/prajwal/Projects/CancerDetector/normalizedData/TrainData.csv')
    dataset = np.array(dataset)
    target = np.array(target)
    # print(target)
    # thresholds = [0, 0.00001, 0.0002, 0.0005, 0.005, 0.05, 0.5, 0.95, 0.96, 0.9999, 1]
    thresholds = [0, 0.3, 0.5, 0.7, 0.8, 1]
    output = []
    for row in dataset:
        output.append(forwardPropagate(network, row))
    TPR = []
    FPR = []
    for cut_off in thresholds:
        print(cut_off)
        result = []
        for row in output:
            if row[0] > cut_off:
                result.append([1])
            else:
                result.append([0])
        result = np.array(result)
        # print(result)
        confusion = confusionMatrix(target, result)
        tpr, fpr = evaluateModel(confusion)
        TPR.append(tpr)
        FPR.append(fpr)
    print('True Positive Rate', TPR, '\n',
          'False Positive Rate', FPR)
    ROC_Curve(TPR, FPR)
