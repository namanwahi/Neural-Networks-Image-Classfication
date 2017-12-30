#training and testing a linear model
import numpy as np
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data

#seed RNG for demonstration purposes (can be removed)
np.random.seed(0)


(X_train, y_train), (X_test, y_test) = get_imagesf()
mean_image = preprocess_training_data(X_train)

input_size = X_train.shape[1]
N = X_train.shape[0]

eta = 1          # learning rate
n_epochs = 200   # number of epochs
class_no = 10    # number of classes


W = 0.01 * np.random.randn(input_size, class_no) #weights
b = np.zeros(class_no).T                         #biases
                           
for i in range(n_epochs):
    #forward pass whole dataset
    all_scores = np.dot(X_train, W) + b

    #eponentiate all the scores
    exp_scores = np.exp(all_scores)

    #calculate cross entropy loss
    probs  = exp_scores / np.sum(exp_scores, axis =1, keepdims=True)
    total_loss = np.sum(-np.log(probs[range(N), y_train]))
    avg_loss = total_loss / N

    print("Epoch: " + str(i)  + " Average loss: " + str(avg_loss))

    #derivative of loss function with respect to the scores
    dscores = probs
    dscores[range(N), y_train] = dscores[range(N), y_train] - 1
    dscores = dscores / N

    #derivative of the loss function with respect to the weights/biases
    dW = np.dot(X_train.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    #update the weights/biases
    W = W - (eta * dW)
    b = b - (eta * db)

    
#preprocess the test data
preprocess_test_data(X_test, mean_image)

scores = np.dot(X_test, W) + b

#use the index of the max score as the predicted class
predictions = np.argmax(scores, axis = 1)
print("Overall Accuracy " + str(np.mean(predictions == y_test)*100) + '%')

for i in range(class_no):
    indices = np.where(i == y_test)
    print("Accuract of class " + str(i) + ": " + str(np.mean(predictions[indices] == y_test[indices])*100) + '%')

    
