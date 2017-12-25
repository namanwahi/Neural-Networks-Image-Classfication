import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

def get_color_and_label(class_index):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    class_labels = [ "T-shirt/top - 0", "Trouser - 1", "Pullover - 2",  "Dress - 3", "Coat - 4", "Sandal - 5", "Shirt - 6", "Sneaker - 7", "Bag - 8", "Ankle boot - 9"]
    return colors[class_index], class_labels[class_index]

from reader import get_images
(X_train, y_train), _ = get_images() 

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_train)

fig, plot = plt.subplots()
fig.set_size_inches(25, 25)

#color_array = [get_color_and_label(y_train[i])[0] for i in range(len(X_transformed))]
#label_array = [get_color_and_label(y_train[i])[1] for i in range(len(X_transformed))]
#plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train, label=y_train)

class_no = 10
for class_ in range(10):
    indices = np.where(y_train == class_)
    plot.scatter(X_transformed[indices, 0], X_transformed[indices, 1], label=class_)
    
plt.legend(fontsize = 30)


plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.xlabel('PCA axis 1', fontsize = 40)
plt.ylabel('PCA axis 2', fontsize = 40)


plt.tight_layout()
plt.savefig("FasionMNISTPCA.png")
