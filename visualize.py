import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
d = sio.loadmat('/home/huy/Github/idnns/data/var_u.mat')
X = d['F'] # (n_samples, n_feature)
y = np.squeeze(d['y'].T, axis=1)

# Create a PCA instance with two components
pca = PCA(n_components=2, random_state=2023)

# Fit the data to the PCA model
X_pca = pca.fit_transform(X)

# Visualize the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig(f'visualize.png', dpi=400)

from mpl_toolkits.mplot3d import Axes3D

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a PCA instance with two components
pca = PCA(n_components=3, random_state=0 )

# Fit the data to the PCA model
X_pca = pca.fit_transform(X)

scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis')

# Set the axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Set the colorbar and its label
fig.colorbar(scatter)

# Show the plot
plt.savefig(f'visualize.png', dpi=400)