import os
import sys
# from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import idx2numpy
import scipy.io as sio

import torch
from torch.utils.data import Dataset, DataLoader

# Custom data
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.FloatTensor)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

def load_data(name, random_labels=False):
	"""Load the data
	name - the name of the dataset
	random_labels - True if we want to return random labels to the dataset
	return object with data and labels"""
	print ('Loading Data...')
	C = type('type_C', (object,), {})
	data_sets = C()
	if name.split('/')[-1] == 'MNIST':
		train_images_file = os.path.dirname(sys.argv[0]) + "data/MNIST_data/train-images-idx3-ubyte"
		train_label_file = os.path.dirname(sys.argv[0]) + "data/MNIST_data/train-labels-idx1-ubyte"
		test_images_file = os.path.dirname(sys.argv[0]) + "data/MNIST_data/t10k-images-idx3-ubyte"
		test_label_file = os.path.dirname(sys.argv[0]) + "data/MNIST_data/t10k-labels-idx1-ubyte"
		
		train_images = idx2numpy.convert_from_file(train_images_file)
		train_labels = idx2numpy.convert_from_file(train_label_file)
		test_images = idx2numpy.convert_from_file(test_images_file)
		test_labels = idx2numpy.convert_from_file(test_label_file)

		# data_sets_temp = input_data.read_data_sets(os.path.dirname(sys.argv[0]) + "/data/MNIST_data/", one_hot=True)
		data_sets.data = np.concatenate((train_images, test_images), axis=0)
		data_sets.labels = np.concatenate((train_labels, test_labels), axis=0)
		data_sets.data = np.reshape(data_sets.data, (-1, 784))
		data_sets.labels = np.eye(10)[data_sets.labels]
	else:
		d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
		F = d['F']
		y = d['y']
		C = type('type_C', (object,), {})
		data_sets = C()
		data_sets.data = F
		data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)

	# If we want to assign random labels to the  data
	if random_labels:
		labels = np.zeros(data_sets.labels.shape)
		labels_index = np.random.randint(low=0, high=labels.shape[1], size=labels.shape[0])
		labels[np.arange(len(labels)), labels_index] = 1
		data_sets.labels = labels
	return data_sets


def load_data_numpy(name, random_labels=False):
	"""
	This function is the same as load_data() but return a dictionary of numpy arrays instead of "type_C" type
	"""
	print ('Loading Data...')
	datasets = {}
	if name.split('/')[-1] == 'MNIST':
		train_images_file = os.path.dirname(sys.argv[0]) + "data/MNIST_data/train-images-idx3-ubyte"
		train_label_file = os.path.dirname(sys.argv[0]) + "data/MNIST_data/train-labels-idx1-ubyte"
		test_images_file = os.path.dirname(sys.argv[0]) + "data/MNIST_data/t10k-images-idx3-ubyte"
		test_label_file = os.path.dirname(sys.argv[0]) + "data/MNIST_data/t10k-labels-idx1-ubyte"
		
		train_images = idx2numpy.convert_from_file(train_images_file)
		train_labels = idx2numpy.convert_from_file(train_label_file)
		test_images = idx2numpy.convert_from_file(test_images_file)
		test_labels = idx2numpy.convert_from_file(test_label_file)

		data = np.concatenate((train_images, test_images), axis=0)
		labels = np.concatenate((train_labels, test_labels), axis=0)

		data = np.reshape(data, (-1, 784))
		labels = np.eye(10)[labels]
	else:
		d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + '.mat'))
		F = d['F'].astype(np.float32)
		y = d['y'].astype(np.float32)

		data = F
		labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)

	# If we want to assign random labels to the  data
	if random_labels:
		labels = np.zeros(labels.shape)
		labels_index = np.random.randint(low=0, high=labels.shape[1], size=labels.shape[0])
		labels[np.arange(len(labels)), labels_index] = 1
	
	datasets['data'] = data
	datasets['labels'] = labels

	return datasets


def shuffle_in_unison_inplace(a, b):
	"""Shuffle the arrays randomly"""
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
	"""Divided the data to train and test and shuffle it"""
	perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
	print('perc', perc)
	print('percent_of_train', percent_of_train)
	C = type('type_C', (object,), {})
	data_sets = C()
	stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
	start_test_index = stop_train_index
	if percent_of_train > min_test_data:
		start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
	data_sets.train = C()
	data_sets.test = C()
	if shuffle_data:
		shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
	else:
		shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
	data_sets.train.data = shuffled_data[:stop_train_index[0], :]
	data_sets.train.labels = shuffled_labels[:stop_train_index[0], :]
	data_sets.test.data = shuffled_data[start_test_index[0]:, :]
	data_sets.test.labels = shuffled_labels[start_test_index[0]:, :]
	return data_sets


def data_shuffle_pytorch(data_sets_org, percent_of_train, batch_size, min_test_data=80, shuffle_data=False):
	"""Divided the data to train and test and shuffle it"""
	input_size = data_sets_org['data'].shape[1]
	num_of_classes = len(np.unique(data_sets_org['labels'], axis=0))
	perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
	print('perc', perc)
	print('percent_of_train', percent_of_train)
	stop_train_index = perc(percent_of_train[0], data_sets_org['data'].shape[0])
	start_test_index = stop_train_index
	if percent_of_train > min_test_data:
		start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
	if shuffle_data:
		shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org['data'], data_sets_org['labels'])
	else:
		shuffled_data, shuffled_labels = data_sets_org['data'], data_sets_org['labels']

	X, y = data_sets_org['data'], data_sets_org['labels']

	X_train, y_train = shuffled_data[:stop_train_index[0], :], shuffled_labels[:stop_train_index[0]]
	X_test, y_test = shuffled_data[start_test_index[0]:, :], shuffled_labels[start_test_index[0]:]

	all_data_dataset = CustomDataset(X, y)
	train_dataset = CustomDataset(X_train, y_train)
	test_dataset = CustomDataset(X_test, y_test)

	all_data_dataloader = DataLoader(all_data_dataset, batch_size=batch_size, shuffle=False)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	return train_dataloader, test_dataloader, all_data_dataloader, input_size, num_of_classes
