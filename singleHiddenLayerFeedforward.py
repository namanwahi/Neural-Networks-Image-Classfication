import numpy as np
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data

import torch
from torch.autograd import Variable

#refactor conversion to float64
_, (X_test, y_test) = get_imagesf()

model = torch.load("FeedforwardNN.pt")
X_test = Variable(torch.from_numpy(X_test).float())
y_test = Variable(torch.from_numpy(y_test))

scores = model(X_test)
_, predictions = torch.max(scores.data, 1)
mean = torch.mean((predictions.byte() == y_test.data).float())
print(mean)

