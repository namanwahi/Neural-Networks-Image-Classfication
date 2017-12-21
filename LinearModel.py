import numpy as np
from reader import get_imagesf
from dataPreprocessing import  preprocess_training_data,  preprocess_test_data

#refactor conversion to float64
(X_train, y_train), (X_test, y_test) = get_imagesf()

mean_image = preprocess_training_data(X_train)

class_no = 10
input_size = 28 ** 2
N = X_train.shape[0]

eta = 1        # learning rate
n_epochs = 200   # number of epochs

W = 0.01 * np.random.randn(input_size, class_no)
b = np.zeros((1, class_no))
                           
for i in range(n_epochs):
    #forward pass whole 
    all_scores = np.dot(X_train, W) + b

    exp_scores = np.exp(all_scores)

    #refactor
    probs  = exp_scores / np.sum(exp_scores, axis =1 , keepdims=True)

    
    total_loss = -np.log(probs[range(N), y_train])
    avg_loss = np.sum(total_loss) / N

    print("epoch: " + str(i)  + " " + str(avg_loss))

    # Backward pass: compute gradients
    dscores = probs
    dscores[range(N), y_train] = dscores[range(N), y_train] - 1
    dscores = dscores / N

    # Compute gradients
    dW = np.dot(X_train.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    # Update rule: Stochastic Gradient Descent
    W = W - (eta * dW)
    b = b - (eta * db)

    
#test accuracy

#preprocess the test data
preprocess_test_data(X_test, mean_image)
scores = np.dot(X_test, W) + b
predictions = np.argmax(scores, axis = 1)
print(np.mean(predictions == y_test))


