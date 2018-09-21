from __future__ import print_function
import numpy as np
from PreProcessing.InitilizeRBM import initWeights
from PreProcessing.TrainRBMTest import trainRBM


def deepBeliefNet(normalized_data):
    train_data = normalized_data
    num_visible1 = train_data.shape[1]
    num_hidden1 = 25
    weights = initWeights(num_visible=num_visible1, num_hidden=num_hidden1)
    # print('initilized weight\n', weights)
    name = generate_name(num_hidden1)
    columns = ','.join(name)
    # columns = ','.join(['bias', 'hidden1', 'hidden2', 'hidden3', 'hidden4', 'hidden5', 'hidden6'])
    rows = ['bias', 'visible1', 'visible2', 'visible3', 'visible4', 'visible5', 'visible6', 'visible7', 'visible8']
    np.savetxt('/home/prajwal/Projects/CancerDetector/weights/initialWeight.csv', weights, delimiter=',', header=columns, comments='')

    trained1 = trainRBM(train_data, weights=weights, num_hidden=num_hidden1)
    # print('weight_RBM1\n', trained1[0])

    name = generate_name(num_hidden1)
    columns = ','.join(name)
    # columns = ','.join(['bias', 'hidden1', 'hidden2', 'hidden3', 'hidden4', 'hidden5', 'hidden6'])
    rows = ['bias', 'visible1', 'visible2', 'visible3', 'visible4', 'visible5', 'visible6', 'visible7', 'visible8']
    np.savetxt('/home/prajwal/Projects/CancerDetector/weights/RBM1.csv', trained1[0], delimiter=',', header=columns, comments='')

    num_visible2 = 25
    num_hidden2 = 10
    weights2 = initWeights(num_visible=num_visible2, num_hidden=num_hidden2)
    train_data2 = np.delete(trained1[1], 0, 1)
    trained2 = trainRBM(train_data2, weights=weights2, num_hidden=num_hidden2)
    # print('weight_RBM2\n', trained2[0])

    name = generate_name(num_hidden2)
    columns = ','.join(name)
    # columns = ','.join(['bias', 'hidden1', 'hidden2', 'hidden3', 'hidden4'])
    rows = ['bias', 'visible1', 'visible2', 'visible3', 'visible4', 'visible5', 'visible6']
    np.savetxt('/home/prajwal/Projects/CancerDetector/weights/RBM2.csv', trained2[0], delimiter=',', header=columns, comments='')


    num_visible3 = 10
    num_hidden3 = 10
    weights3 = initWeights(num_visible=num_visible3, num_hidden=num_hidden3)
    train_data3 = np.delete(trained2[1], 0, 1)
    trained3 = trainRBM(train_data3, weights=weights3, num_hidden=num_hidden3)
    # print('Weight_RBM3\n', trained3[0])

    name = generate_name(num_hidden3)
    columns = ','.join(name)
    # columns = ','.join(['bias', 'hidden1', 'hidden2'])
    rows = ['bias', 'visible1', 'visible2', 'visible3', 'visible4']
    np.savetxt('/home/prajwal/Projects/CancerDetector/weights/RBM3.csv', trained3[0], delimiter=',', header=columns, comments='')


    num_visible4 = 10
    num_hidden4 = 1
    weights4 = initWeights(num_visible=num_visible4, num_hidden=num_hidden4)
    train_data4 = np.delete(trained3[1], 0, 1)
    trained4 = trainRBM(train_data4, weights=weights4, num_hidden=num_hidden4)
    # print('Weight_RBM4\n', trained4[0])

    name = generate_name(num_hidden4)
    columns = ','.join(name)
    # columns = ','.join(['bias', 'hidden'])
    rows = ['bias', 'visible1', 'visible2']
    np.savetxt('/home/prajwal/Projects/CancerDetector/weights/RBM4.csv', trained4[0], delimiter=',', header=columns, comments='')


def generate_name(num_hidden):
    name = []
    for i in range(num_hidden + 1):
        name.append('hidden{}'.format(i))
    return name