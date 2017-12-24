import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

def get_color_from_class_index(class_index):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    return colors[class_index]
    
from reader import get_images
(X_train, y_train), _ = get_images() 

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_train)

fig, plot = plt.subplots()
fig.set_size_inches(50, 50)

color_array = [get_color_from_class_index(y_train[i]) for i in range(len(X_transformed))]
plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color_array)

plot.set_xticks(())
plot.set_yticks(())

plt.tight_layout()
plt.savefig("mnist_pca1.png")
