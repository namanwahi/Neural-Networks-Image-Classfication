import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d, relu
from torch.autograd import Variable
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from CNNClasses import ConvNet

def trainCNN(np_X_train, np_y_train):
    """
    Trains a convolutional network given the training data.

    Args
    -----
    np_X_train: a numpy array of preprocessed data with shape (num_samples, 1, width, height)
    np_y_train: a numpy array of the training classes with shape (num_samples,)

    Returns
    -----

    A trained ConvNet object
    """

    #convert to tensors
    X_train = torch.from_numpy(np_X_train).float()
    y_train = torch.from_numpy(np_y_train)

    #create a dataloader with batch size 256
    batch_size = 256
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size = batch_size)

    model = ConvNet()
    
    eta = 1e-2
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = eta)

    n_epochs = 100
    loop_loss  = None
    for i in range(n_epochs):
        for x_b, y_b in loader:
            #for each batch

            #convert to autograd variables
            x_v, y_v = Variable(x_b), Variable(y_b)

            #calculate loss
            y_pred = model(x_v)
            loss = loss_fn(y_pred, y_v)

            #calculate gradients and optimise weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop_loss = loss
        print("epoch " + str(i) + " loss:" + str(loop_loss))
    return model

(X_train, y_train), (X_test, y_test)  = get_imagesf()

#preprocess whole training data set
mean_image = Variable(torch.from_numpy(preprocess_training_data(X_train_full)).float())

#reshape to pass to conv net
X_train = np.reshape(X_train, (-1, 1, 28, 28))

#serialise the model
model = trainCNN(X_train, y_train)
torch.save(model, "CNN.pt")
