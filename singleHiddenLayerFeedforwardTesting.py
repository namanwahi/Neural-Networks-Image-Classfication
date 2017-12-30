import numpy as np
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data

import torch
from torch.autograd import Variable

_, (X_test, y_test) = get_imagesf()

#load the trained neural network
model = torch.load("singleHiddenLayerANN.pt")

#convert to autograd variables from numpy arrays
X_test = Variable(torch.from_numpy(X_test).float())
y_test = Variable(torch.from_numpy(y_test))

#pass the test data through the neural network
scores = model(X_test)

#use the max value of each set of scores as the predicted class
_, predictions = torch.max(scores.data, 1)

#overall performance
y_test, predictions = y_test.data.numpy(), predictions.numpy()
print("Overall Accuracy: " + str(np.mean(predictions == y_test)*100) + '%')

#performance of each class
class_no = 10
for i in range(class_no):
    indices = np.where(i == y_test)
    print("Accuracy of class " + str(i) + ": " + str(np.mean(predictions[indices] == y_test[indices])*100) + '%')



