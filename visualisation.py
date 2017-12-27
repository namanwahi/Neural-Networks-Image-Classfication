import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dataPreprocessing import preprocess_training_data

from reader import get_imagesf
(X_train, y_train), _ = get_imagesf() 
preprocess_training_data(X_train)

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_train)

fig, plot = plt.subplots()
fig.set_size_inches(25, 25)

class_labels = [ "T-shirt/top - 0", "Trouser - 1", "Pullover - 2",  "Dress - 3", "Coat - 4", "Sandal - 5", "Shirt - 6", "Sneaker - 7", "Bag - 8", "Ankle boot - 9"]
for class_index in range(10):
    indices = np.where(y_train == class_index)
    plot.scatter(X_transformed[indices, 0], X_transformed[indices, 1], label=class_labels[class_index])
    
plt.legend(fontsize = 20)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)

plt.xlabel('PCA axis 1', fontsize = 40)
plt.ylabel('PCA axis 2', fontsize = 40)


#plt.tight_layout()
plt.savefig("FasionMNISTPCA.png")
