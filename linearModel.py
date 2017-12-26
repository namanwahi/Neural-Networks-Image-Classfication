import numpy as np
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data

np.random.seed(0)

(X_train, y_train), (X_test, y_test) = get_imagesf()

mean_image = preprocess_training_data(X_train)

input_size = X_train.shape[1]
N = X_train.shape[0]

eta = 1          # learning rate
n_epochs = 200   # number of epochs

class_no = 10

W = 0.01 * np.random.randn(input_size, class_no)
b = np.zeros(class_no).T
                           
for i in range(n_epochs):
    #forward pass whole 
    all_scores = np.dot(X_train, W) + b

    exp_scores = np.exp(all_scores)

    probs  = exp_scores / np.sum(exp_scores, axis =1, keepdims=True)
    
    total_loss = np.sum(-np.log(probs[range(N), y_train]))
    avg_loss = total_loss / N

    print("Epoch: " + str(i)  + " Average loss: " + str(avg_loss))
    
    dscores = probs
    dscores[range(N), y_train] = dscores[range(N), y_train] - 1
    dscores = dscores / N
    
    dW = np.dot(X_train.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    
    W = W - (eta * dW)
    b = b - (eta * db)

    

#preprocess the test data
preprocess_test_data(X_test, mean_image)
scores = np.dot(X_test, W) + b
predictions = np.argmax(scores, axis = 1)
print("Overall Accuracy " + str(np.mean(predictions == y_test)*100) + '%')

for i in range(class_no):
    indices = np.where(i == y_test)
    print("Accuract of class " + str(i) + ": " + str(np.mean(predictions[indices] == y_test[indices])*100) + '%')

    
