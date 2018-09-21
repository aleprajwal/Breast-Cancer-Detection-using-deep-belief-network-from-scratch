import numpy as np
from random import randrange
from PreProcessing.Activation import sigmoid


def trainRBM(train_data, weights, num_hidden, max_epoch=700, learning_rate=0.001):
    num_examples = train_data.shape[0]

    # Insert bias units of 1 into the first column.
    train_data = np.insert(train_data, 0, 1, axis=1)

    for epoch in range(max_epoch):
        # (This is the "positive CD phase", aka the reality phase.)
        pos_hidden_activations = np.dot(train_data, weights)
        pos_hidden_probs = sigmoid(pos_hidden_activations)
        pos_hidden_probs[:, 0] = 1
        pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, num_hidden + 1)
        pos_associations = np.dot(train_data.T, pos_hidden_probs)

        # (This is the "negative CD phase", aka the daydreaming phase.)
        neg_visible_activations = np.dot(pos_hidden_states, weights.T)
        neg_visible_probs = sigmoid(neg_visible_activations)
        neg_visible_probs[:, 0] = 1

        neg_hidden_activations = np.dot(neg_visible_probs, weights)
        neg_hidden_probs = sigmoid(neg_hidden_activations)
        neg_hidden_probs[:, 0] = 1

        neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

        # Update weights.
        weights += learning_rate * ((pos_associations - neg_associations) / num_examples)

        error = np.sum((train_data - neg_visible_probs) ** 2)
        if True:
            print("Epoch {}s: error is {}s" .format(epoch, error))
    print("\n\n-----**** RBM Trained ***----\n\n\n\n")
    # print("Epoch %s: error is %s" % (epoch, error))

    return weights, neg_hidden_probs

