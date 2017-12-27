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

y_test, predictions = y_test.data.numpy(), predictions.numpy()
print("Overall Accuracy " + str(np.mean(predictions == y_test)*100) + '%')

class_no = 10
for i in range(class_no):
    indices = np.where(i == y_test)
    print("Accuract of class " + str(i) + ": " + str(np.mean(predictions[indices] == y_test[indices])*100) + '%')



