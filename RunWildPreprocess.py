from BasicTool.LoadDataset import loadDataset
from BasicTool.Normalization import normalizeDataset
# from PreProcessing.DBN import deepBeliefNet
from PreProcessing.DeepBeliefNet import deepBeliefNet
import numpy as np
import time

start = time.time()

dataset, _ = loadDataset('/home/prajwal/Projects/CancerDetector/normalizedData/TrainData.csv')
# normalize = normalizeDataset(dataset)
# deeBeeN(normalize)

dataset = np.array(dataset)
deepBeliefNet(dataset)
end = time.time() - start
print("\t\tTraining Time : {} seconds\n\n".format(end))
