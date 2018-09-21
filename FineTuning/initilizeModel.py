import pandas as pd
import pickle


def init():
    paths = ['/home/prajwal/Projects/CancerDetector/weights/RBM1.csv',
             '/home/prajwal/Projects/CancerDetector/weights/RBM2.csv',
             '/home/prajwal/Projects/CancerDetector/weights/RBM3.csv',
             '/home/prajwal/Projects/CancerDetector/weights/RBM4.csv']

    network = list()
    for path in paths:
        df = pd.read_csv(path)
        dframe = df.drop(columns='hidden0')
        # print(dframe)

        neurons = list()
        for i in range(len(dframe.columns)):
            neuron = dframe.iloc[:, i]
            # print(neuron)
            neurons.append({'weights': list(neuron)})

        # print(neurons)
        network.append(neurons)

    # print(network)
    return network


if __name__ == '__main__':
    network = init()
    # for row in network:
    #     print(row)
    filename = '/home/prajwal/Projects/CancerDetector/classifier/experiments/preTrainedModel_05.pkl'
    pickle.dump(network, open(filename, 'wb'))
    print(network)