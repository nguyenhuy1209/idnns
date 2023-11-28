import numpy as np
import idx2numpy
import seaborn as sns
import matplotlib.pyplot as plt


file = 'data/MNIST_data/train-images-idx3-ubyte'

arr = idx2numpy.convert_from_file(file)
shuffled_indices = np.arange(len(arr))
np.random.shuffle(shuffled_indices)

arr = arr[shuffled_indices]
arr = arr[:10000]
arr = arr.flatten()
print(arr.shape)
print(np.unique(arr))
sns.countplot(data=arr, palette='viridis')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Categorical Data Distribution')
plt.xticks(rotation=45)  # Rotate the category labels for better visibility
plt.savefig('data_hist.png')
