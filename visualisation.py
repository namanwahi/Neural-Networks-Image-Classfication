import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata

#use all digits
#mnist = fetch_mldata("MNIST original")
#X_train, y_train = mnist.data[:70000], mnist.target[:70000]

#X_train, y_train = shuffle(X_train, y_train)
#X_train, y_train = X_train[:1000], y_train[:1000]  # lets subsample a bit for a first impression

from reader import get_images
(X_train, y_train), (x_test, y_test) = get_images() 

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X_train)

fig, plot = plt.subplots()
fig.set_size_inches(50, 50)
plt.prism()

zeroToNine = np.arange(2)
cycle = np.repeat(zeroToNine, 30000)

plot.scatter(X_transformed[:, 0], X_transformed[:, 1], c=cycle)
plot.set_xticks(())
plot.set_yticks(())

plt.tight_layout()
plt.savefig("mnist_pca1.png")
