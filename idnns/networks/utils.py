import os
import sys

import idx2numpy
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

np.random.seed(0)
np.set_printoptions(threshold=sys.maxsize)


# Custom data
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.FloatTensor)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data_numpy(name, random_labels=False):
    """
    This function is the same as load_data() but return a dictionary of numpy arrays instead of "type_C" type
    """
    print("Loading Data...")
    datasets = {}
    if name.split("/")[-1] == "housing":
        # Load the dataset
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.DataFrame(data.target, columns=["MedianHouseValue"])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Convert to PyTorch tensors
        X_train = X_train_scaled
        y_train = y_train.values
        X_test = X_test_scaled
        y_test = y_test.values

        datasets["train_data"] = X_train
        datasets["train_label"] = y_train
        datasets["test_data"] = X_test
        datasets["test_label"] = y_test
        datasets["sample_data"] = np.concatenate((X_train, X_test), axis=0)
        datasets["sample_label"] = np.concatenate((y_train, y_test), axis=0)
    elif name.split("/")[-1] == "MNIST":
        train_images_file = (
            os.path.dirname(sys.argv[0]) + "data/MNIST/train-images-idx3-ubyte"
        )
        train_label_file = (
            os.path.dirname(sys.argv[0]) + "data/MNIST/train-labels-idx1-ubyte"
        )
        test_images_file = (
            os.path.dirname(sys.argv[0]) + "data/MNIST/t10k-images-idx3-ubyte"
        )
        test_label_file = (
            os.path.dirname(sys.argv[0]) + "data/MNIST/t10k-labels-idx1-ubyte"
        )

        train_images = idx2numpy.convert_from_file(train_images_file)
        train_labels = idx2numpy.convert_from_file(train_label_file)
        test_images = idx2numpy.convert_from_file(test_images_file)
        test_labels = idx2numpy.convert_from_file(test_label_file)

        train_images = np.reshape(train_images, (-1, 784))
        train_labels = np.eye(10)[train_labels]
        test_images = np.reshape(test_images, (-1, 784))
        test_labels = np.eye(10)[test_labels]

        datasets["train_data"] = train_images
        datasets["train_label"] = train_labels
        datasets["test_data"] = test_images
        datasets["test_label"] = test_labels

        half_samples = train_images.shape[0] // 2
        indices = np.random.choice(train_images.shape[0], half_samples, replace=False)
        train_images = train_images[indices, :]
        train_labels = train_labels[indices, :]
        datasets["sample_data"] = np.concatenate((train_images, test_images), axis=0)
        datasets["sample_label"] = np.concatenate((train_labels, test_labels), axis=0)
    else:
        d = sio.loadmat(os.path.join(os.path.dirname(sys.argv[0]), name + ".mat"))
        F = d["F"].astype(np.float32)
        y = d["y"].astype(np.float32)

        data = F
        labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)

        # If we want to assign random labels to the  data
        if random_labels:
            labels = np.zeros(labels.shape)
            labels_index = np.random.randint(
                low=0, high=labels.shape[1], size=labels.shape[0]
            )
            labels[np.arange(len(labels)), labels_index] = 1

        datasets["data"] = data
        datasets["labels"] = labels

    return datasets


def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train, min_test_data=80, shuffle_data=False):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    print("perc", perc)
    print("percent_of_train", percent_of_train)
    C = type("type_C", (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
    start_test_index = stop_train_index
    if percent_of_train > min_test_data:
        start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(
            data_sets_org.data, data_sets_org.labels
        )
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[: stop_train_index[0], :]
    data_sets.train.labels = shuffled_labels[: stop_train_index[0], :]
    data_sets.test.data = shuffled_data[start_test_index[0] :, :]
    data_sets.test.labels = shuffled_labels[start_test_index[0] :, :]
    return data_sets


def data_shuffle_pytorch(
    data_sets_org,
    percent_of_train,
    batch_size,
    min_test_data=80,
    shuffle_data=False,
    is_mnist=False,
    is_regresion=False,
):
    """Divided the data to train and test and shuffle it"""
    if is_mnist or is_regresion:
        input_size = data_sets_org["train_data"].shape[1]
        if is_regresion:
            num_of_classes = 1
        else:
            num_of_classes = len(np.unique(data_sets_org["train_label"], axis=0))

        X_train, y_train = (
            data_sets_org["train_data"],
            data_sets_org["train_label"],
        )
        X_test, y_test = (
            data_sets_org["test_data"],
            data_sets_org["test_label"],
        )

        X = data_sets_org["sample_data"]
        y = data_sets_org["sample_label"]

    else:
        input_size = data_sets_org["data"].shape[1]
        num_of_classes = len(np.unique(data_sets_org["labels"], axis=0))
        perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
        print("perc", perc)
        print("percent_of_train", percent_of_train)
        stop_train_index = perc(percent_of_train[0], data_sets_org["data"].shape[0])
        start_test_index = stop_train_index
        if percent_of_train > min_test_data:
            start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
        if shuffle_data:
            shuffled_data, shuffled_labels = shuffle_in_unison_inplace(
                data_sets_org["data"], data_sets_org["labels"]
            )
        else:
            shuffled_data, shuffled_labels = (
                data_sets_org["data"],
                data_sets_org["labels"],
            )

        X, y = data_sets_org["data"], data_sets_org["labels"]

        X_train, y_train = (
            shuffled_data[: stop_train_index[0], :],
            shuffled_labels[: stop_train_index[0]],
        )
        X_test, y_test = (
            shuffled_data[start_test_index[0] :, :],
            shuffled_labels[start_test_index[0] :],
        )

    all_data_dataset = CustomDataset(X, y)
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    all_data_dataloader = DataLoader(
        all_data_dataset, batch_size=batch_size, shuffle=False
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_dataloader,
        test_dataloader,
        all_data_dataloader,
        input_size,
        num_of_classes,
    )
