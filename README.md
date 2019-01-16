# Breast-Cancer-Detection-using-deep-belief-network-from-scratch
DBN (deep belief network) implementation for breast cancer detection

In this project i've used Deep belief network (DBN) to pre-train the model. Then, i've used back-propagation algorithm to optimize the model. This project is completely developed from scratch using basic library like numpy, pandas and matplotlib.
Dataset is taken from UCI Machine Learning Repository 

Link to dataset: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

FOLDERS DETAILS

	1. Basic Tool
		-> LoadDataset.py ==> load dataset 
 		-> Normalization.py ==> normalize dataset 


	2. classifier
	In this folder trained classifier model is saved.
 
	3. dataset 
	It contains datasets. There four csv files in this folder. 
	The original.csv is original dataset from winsconsin machinelearning repositiory.
		-> csv_db.csv ==> cleaned datasets. It is divided into train and test dataset. (TrainData.csv and TestData.csv)
		-> TrainData.csv ==> for training classifier model
		-> TestData.csv ==> for testing trained classifier model

	4. normalization
	Dataset is stored into this folder after normalization.
		-> normalized.csv
		-> TestData.csv
		-> TrainData.csv

	5. Preprocessing
	Implementing Deep Belief Network (DBN)
		-> Activation.py ==> activation function
		-> DeepBeliefNet.py ==> creating deep belief network 
		-> InitilizationRBM.py ==> initilizating RBM (Restricted Boltzmann Machine)
		-> TrainRBM.py ===> training RBM without creating batch of dataset
		-> TrainRBMTest.py ===> training RBM by creating batch of dataset

	Note: network layers 
  
		Input layer = 9 neurons (8 features and 1 bias)
		1st hidden layer => 25 neurons
		2nd hidden layer => 25 neurons
		3rd hidden layer => 10 neurons
		output layer => 1 neuron


	6. python
	this folder is created to provide user service using UI. I used web based UI but I have not mentioned here. 
	two folders: 
		1. classifier 
			-> fineTunedModel.pkl	 ===> trained model to classify new data’s

		2. json 	
			-> data.json ==> data extracted from user
			-> result.json ==> result after classification


	7. weights
	weights from Pre-Training 
		-> initialWeight.csv
		-> RBM1.csv ==> weights between input and 1st hidden layer
		-> RBM2.csv ==> weight between 1st and 2nd hidden layer
		-> RBM3.csv ==> weight between 2nd and 3rd hidden layer 
		-> RBM4.csv ==> weight between 3rd and output layer
