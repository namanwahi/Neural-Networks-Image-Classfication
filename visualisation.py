import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

from reader import get_images
(X_train, y_train), (x_test, y_test) = get_images() 

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_train)

fig, plot = plt.subplots()
fig.set_size_inches(50, 50)
plt.prism()


plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_train)
plot.set_xticks(())
plot.set_yticks(())

plt.tight_layout()
plt.savefig("mnist_pca1.png")
