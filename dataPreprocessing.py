import numpy as np

def scale_pixels(data):
    """
    Scales the range of the pixel values from [0, 255] to [0, 1]

    Args
    -------
    data : an array object with values in range [0, 255]
    """
    data /= 255
    
def preprocess_training_data(training_data):
    """
    Scales and zero-centres the training data and returns the mean image.

    Args
    -------
    training_data : an array with shape (num_samples, pixel_count) with values in the range [0, 255]

    Returns
    -------
    The mean image of the training dataset with shape (pixel_count,)
    """
    scale_pixels(training_data)

    #zero centre the data
    mean_image = np.mean(training_data, axis=0)
    training_data -= mean_image
    return mean_image

def preprocess_test_data(test_data, mean_image):
    """
    Scales and zero-centres the test_data using the mean image of its training dataset.
    
    Args
    -------
    test_data : an array with shape (num_samples, 784) with values in the range [0, 255]
    mean_imge: an array with shape (pixel_count,) of the mean image of the training data

    """
    scale_pixels(test_data)
    #zero centre the data using the mean_image
    test_data -= mean_image
