#methods for training a single hidden layer neural network
import numpy as np
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from random import randint, uniform
import torch.nn as nn


def train_NN(np_X_train, np_y_train, H, eta):
    """
    Trains a single hidden layer neural network given some hyperparameters and its training data.

    Args
    -------
    np_X_train: a numpy array with the preprocessed training data images
    np_y_train: a numpy array with the training data classes
    H: the number of neurons in the hidden layer
    eta: the initial learning rate

    Returns
    -------
    A nn.Module of the neural network
    """
    
    X_train = torch.from_numpy(np_X_train).float()
    y_train = torch.from_numpy(np_y_train)

    #create a dataloader to deliver batches of size 256
    batch_size = 256
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size = batch_size)

    #the size of the inputs and outputs of the neural network
    I = np_X_train.shape[1]
    O = 10

    model = torch.nn.Sequential(torch.nn.Linear(I, H), torch.nn.ReLU(), torch.nn.Linear(H, O))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    n_epochs = 100
    epoch_loss  = None
    for i in range(n_epochs):
        for X_batch, y_batch in loader:
            #for each batch
            
            X_batch, y_batch = Variable(X_batch), Variable(y_batch)

            #predict the results and calculate the loss
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            #calculate gradients and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = loss
        print("Loss at last batch of Epoch:" + str(i) + " Average loss:" + str(loop_loss.data[0]))
    return model

def hyperparameter_search(iterations, hidden_l, hidden_u, eta_l, eta_u):
    """
    Searches hyperparameters in the given ranges and logs the result to a file

    Args
    -------
    iterations: the number of searches wished to be performed
    hidden_l: the lower bound of the number of hidden layer neurons
    hidden_u: the upper bound of the number of hidden layer neurons
    eta_l: the lower bound of log_10(initial_learning_rate)
    eta_u: the upper bound of log_10(initial_learning_rate)

    """

    #refactor conversion to float64
    (X_train_full, y_train_full), _  = get_imagesf()
    #preprocess whole training data set
    mean_image = preprocess_training_data(X_train_full)

    #use some of the training data as a validation set
    X_train, y_train  = X_train_full[1000:, :], y_train_full[1000:]
    X_val, y_val = X_train_full[:1000, :], y_train_full[:1000]

    #convert to autograd Variables
    X_val = Variable(torch.from_numpy(X_val).float())
    y_val = Variable(torch.from_numpy(y_val))

    fileName = "SingleHiddenLayerHyperparameterSearchLog.txt"
    with open(fileName, 'a') as file:
        file.write('*' * 50 + '\n')

    for i in range(iterations):
        #choose random hyperparameters within the range and train a network using the training dataset
        H = randint(hidden_l, hidden_u)
        eta  = 10 ** uniform(eta_l, eta_u)
        model = train_NN(X_train, y_train, H, eta)

        #calculate the loss of the model using the validation dataset
        y_pred = model(X_val)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y_val.long())

        #log the results
        message = "Hidden layer units: " + str(H) + "  Initial learning rate: " + str(eta) + " Loss: " + str(loss.data[0]) + '\n'
        with open(fileName, 'a') as file:
            file.write(message)


class FeedForwardNN(nn.Module):
    """
    Class to wrap a model to preprocess the data first before passing it to the model

    Members
    -------
    mean_image: the mean image to be subtracted from the data as part of the preprocessing

    """
    def __init__(self, mean_image, model):
        super(FeedForwardNN, self).__init__()
        self.mean_image = mean_image
        self.model = model

    def forward(self, x):
        x /= 255.0
        x -= self.mean_image
        return self.model(x)
    

def save_model(hidden_layer_size, eta):
    """
    Trains a model using the given hyperparameters and saves the model wrapped with the FeedForwardNN class

    Args
    -------
    hidden_layer_size: the number of neurons in the hidden layer
    eta: the initial learning rate
    """
    (X_train_full, y_train_full), (X_test, y_test)  = get_imagesf()

    #remove the validation set items
    X_train, y_train  = X_train_full[1000:, :], y_train_full[1000:]
    
    #preprocess whole training data set to get the mean image
    mean_image = Variable(torch.from_numpy(preprocess_training_data(X_train)).float())

    #wrap the model and save it
    model = train_NN(X_train, y_train, hidden_layer_size, eta)
    wrapped_model = FeedForwardNN(mean_image, model)
    torch.save(wrapped_model, "singleHiddenLayerANN.pt")
    
