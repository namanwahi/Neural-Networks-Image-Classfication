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
    X_train = torch.from_numpy(np_X_train).float()
    y_train = torch.from_numpy(np_y_train)

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
            x_v, y_v = Variable(x_b), Variable(y_b)
            y_pred = model(x_v)
        
            loss = loss_fn(y_pred, y_v)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop_loss = loss
        print("epoch " + str(i) + " loss:" + str(loop_loss))
    return model

#refactor conversion to float64
(X_train_full, y_train_full), (X_test, y_test)  = get_imagesf()

#preprocess whole training data set
mean_image = Variable(torch.from_numpy(preprocess_training_data(X_train_full)).float())

X_train, y_train  = X_train_full, y_train_full

X_train = np.reshape(X_train, (-1, 1, 28, 28))

#TODO: hyperparameter search
#X_val, y_val = X_train_full[:1000, :], y_train_full[:1000]
model = trainCNN(X_train, y_train)

torch.save(model, "CNN2.pt")
