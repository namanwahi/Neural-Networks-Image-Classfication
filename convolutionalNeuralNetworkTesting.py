import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d, relu
from torch.autograd import Variable
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from CNNClasses import ConvNet

(X_train, y_train), (X_test, y_test)  = get_imagesf()
#preprocess the test data using the mean image of the training data
mean_image = preprocess_training_data(X_train)
preprocess_test_data(X_test, mean_image)

X_test = np.reshape(X_test, (-1, 1, 28, 28))
X_test = Variable(torch.from_numpy(X_test).float())
y_test = Variable(torch.from_numpy(y_test))

#load saved model and pass test data through it
model = torch.load("CNN.pt")
scores = model(X_test)

#overall performance
_, predictions = torch.max(scores.data, 1)
y_test, predictions = y_test, predictions = y_test.data.numpy(), predictions.numpy()
print("Overall Accuracy " + str(np.mean(predictions == y_test)*100) + '%')

#performance of each class
class_no = 10
for i in range(class_no):
    indices = np.where(i == y_test)
    print("Accuract of class " + str(i) + ": " + str(np.mean(predictions[indices] == y_test[indices])*100) + '%')




