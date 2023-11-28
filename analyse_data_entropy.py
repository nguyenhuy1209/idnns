import numpy as np
np.random.seed(0)
import scipy.io as sio
import matplotlib.pyplot as plt

from idnns.information.mutual_information_calculation import calc_multivariate_information, calc_multivariate_information_new

# Read data
data_dir = "data/var_u.mat"
d = sio.loadmat(data_dir)
X = d['F'].astype(np.float32)
y = d['y'].astype(np.float32)
y = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)

# Create an array of shuffled indices
shuffled_indices = np.arange(len(X))
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle both data and labels
X = X[shuffled_indices]
y = y[shuffled_indices]

X = X[:512, :]
y = y[:512, :]

# Search for h
IXXs = []
h = 0.5
alphas = [0.99, 0.999, 1.001, 1.01]
for alpha in alphas:

    m = X.shape[0]
    sigmaX = m ** (-1 / (4 + X.shape[1]))
    sigmaX = h * sigmaX

    # Calculate I(X,X)
    IXX = calc_multivariate_information(X, X, sigmaX, sigmaX, alpha)
    IXXs.append(IXX)

# Draw graph
# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(alphas, IXXs, marker='o', linestyle='-', label='Line Plot', markersize=8, markerfacecolor='r')

# Add labels and a legend
ax.set_xlabel('alphas')
ax.set_ylabel('I(X,X)')
ax.set_title('Testing alpha value from [0.99, 0.999, 1.001, 1.01], h = 0.5')
ax.legend()

ax.set_xticks(alphas)
ax.set_yticks(IXXs)

# Show the plot
plt.savefig('image3.png')

# => choose h = 0.5