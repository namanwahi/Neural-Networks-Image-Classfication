import numpy as np

def scale_pixels(data):
    data /= 255
    
def preprocess_training_data(training_data):
    scale_pixels(training_data)
    #zero centre the data
    mean_image = np.mean(training_data, axis=0)
    training_data -= mean_image
    return mean_image

def preprocess_test_data(test_data, mean_image):
    scale_pixels(test_data)
    #zero centre the data using the mean_image
    test_data -= mean_image
