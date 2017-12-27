import numpy as np
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data

import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from random import randint, uniform
import torch.nn as nn


def train_NN(np_X_train, np_y_train, H, eta):
    X_train = torch.from_numpy(np_X_train).float()
    y_train = torch.from_numpy(np_y_train)

    batch_size = 256
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size = batch_size)

    I = np_X_train.shape[1]
    O = 10

    model = torch.nn.Sequential(
    torch.nn.Linear(I, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, O))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)

    n_epochs = 100
    epoch_loss  = None
    for i in range(n_epochs):
        for X_batch, y_batch in loader:

            X_batch, y_batch = Variable(X_batch), Variable(y_batch)
        
            y_pred = model(X_batch)
        
            loss = loss_fn(y_pred, y_batch)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss = loss
        print("Loss at last batch of Epoch:" + str(i) + " Average loss:" + str(loop_loss.data[0]))
    return model

def hyperparameter_search(iterations, hidden_l, hidden_u, eta_l, eta_u):
    #refactor conversion to float64
    (X_train_full, y_train_full), _  = get_imagesf()
    #preprocess whole training data set
    mean_image = preprocess_training_data(X_train_full)

    X_train, y_train  = X_train_full[1000:, :], y_train_full[1000:]
    X_val, y_val = X_train_full[:1000, :], y_train_full[:1000]

    #convert to tensors
    X_val = Variable(torch.from_numpy(X_val).float())
    y_val = Variable(torch.from_numpy(y_val))

    fileName = "FFANNTesting.txt"
    with open(fileName, 'a') as file:
        file.write('*' * 50 + '\n')

    for i in range(iterations):
        H = randint(hidden_l, hidden_u)
        eta  = 10 ** uniform(eta_l, eta_u)
        model = train_NN(X_train, y_train, H, eta)

        y_pred = model(X_val)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y_val.long())

        message = "Hidden layer units: " + str(H) + "  Initial learning rate: " + str(eta) + " Loss: " + str(loss.data[0]) + '\n'
        with open(fileName, 'a') as file:
            file.write(message)


class FeedForwardNN(nn.Module):
    def __init__(self, mean_image, model):
        super(FeedForwardNN, self).__init__()
        self.mean_image = mean_image
        self.model = model

    def forward(self, x):
        x /= 255.0
        x -= self.mean_image
        return self.model(x)
    

def save_model(hidden_layer_size, eta):
    #refactor conversion to float64
    (X_train_full, y_train_full), (X_test, y_test)  = get_imagesf()

    #preprocess whole training data set
    mean_image = Variable(torch.from_numpy(preprocess_training_data(X_train_full)).float())
    
    X_train, y_train  = X_train_full[1000:, :], y_train_full[1000:]

    model = train_NN(X_train, y_train, hidden_layer_size, eta)

    X_test, y_test = Variable(torch.from_numpy(X_test).float()), Variable(torch.from_numpy(y_test))
    wrapped_model = FeedForwardNN(mean_image, model)
    y_pred = wrapped_model(X_test)

    torch.save(wrapped_model, "FeedforwardNN.pt")
    
