"""Train and calculate the information of network"""

import multiprocessing
import warnings

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from tqdm import tqdm

from idnns.information import information_process as inn
from idnns.networks import model as mo
from idnns.networks.utils import data_shuffle, data_shuffle_pytorch
from idnns.plots.utils import get_data

warnings.filterwarnings("ignore")
summaries_dir = "summaries"
NUM_CORES = multiprocessing.cpu_count()


def build_model(
    activation_function,
    layerSize,
    input_size,
    num_of_classes,
    learning_rate_local,
    save_file,
    covn_net,
):
    """Bulid specipic model of the network
    Return the network model
    """
    model = mo.Model(
        input_size,
        layerSize,
        num_of_classes,
        learning_rate_local,
        save_file,
        int(activation_function),
        cov_net=covn_net,
    )
    return model


def build_model_pytorch(
    activation_function,
    layerSize,
    input_size,
    num_of_classes,
    learning_rate_local,
    save_file,
    covn_net,
):
    """Bulid specipic model of the network
    Return the network model
    """
    model = mo.Model(
        input_size,
        layerSize,
        num_of_classes,
        learning_rate_local,
        save_file,
        int(activation_function),
        cov_net=covn_net,
    )
    return model


def train_and_calc_inf_network(
    i,
    j,
    k,
    layerSize,
    num_of_epochs,
    learning_rate_local,
    batch_size,
    indexes,
    save_grads,
    data_sets_org,
    model_type,
    percent_of_train,
    interval_accuracy_display,
    calc_information,
    calc_information_last,
    num_of_bins,
    interval_information_display,
    save_ws,
    rand_int,
    cov_net,
    is_mnist,
):
    """Train the network and calculate it's information"""
    network_name = "{0}_{1}_{2}_{3}".format(i, j, k, rand_int)
    print("Training network  - {0}".format(network_name))
    network = train_network_pytorch(
        layerSize,
        num_of_epochs,
        learning_rate_local,
        batch_size,
        indexes,
        save_grads,
        data_sets_org,
        model_type,
        percent_of_train,
        interval_accuracy_display,
        network_name,
        cov_net,
        is_mnist,
    )
    network["information"] = []

    if calc_information:
        print("Calculating the information")
        infomration = np.array(
            [
                inn.get_information(
                    network["ws"],
                    data_sets_org["data"],
                    data_sets_org["labels"],
                    num_of_bins,
                    interval_information_display,
                    network["model"],
                    layerSize,
                )
            ]
        )
        network["information"] = infomration
    elif calc_information_last:
        print("Calculating the information for the last epoch")
        infomration = np.array(
            [
                inn.get_information(
                    [network["ws"][-1]],
                    data_sets_org.data,
                    data_sets_org.labels,
                    num_of_bins,
                    interval_information_display,
                    network["model"],
                    layerSize,
                )
            ]
        )
        network["information"] = infomration
    # If we dont want to save layer's output
    if not save_ws:
        network["weights"] = 0
    return network


def exctract_activity(sess, batch_points_all, model, data_sets_org):
    """Get the activation values of the layers for the input"""
    w_temp = []
    for i in range(0, len(batch_points_all) - 1):
        batch_xs = data_sets_org.data[batch_points_all[i] : batch_points_all[i + 1]]
        batch_ys = data_sets_org.labels[batch_points_all[i] : batch_points_all[i + 1]]
        feed_dict_temp = {model.x: batch_xs, model.labels: batch_ys}
        w_temp_local = sess.run([model.hidden_layers], feed_dict=feed_dict_temp)
        # print(len(w_temp_local[0]))
        # print(type(w_temp_local[0][5]))
        # print(w_temp_local[0][0][0])
        # if np.isnan(w_temp_local[0][0]).any():
        # 	print(w_temp_local[0][0])
        # 	print('nan found in weights 0')
        # 	raise
        # if np.isnan(w_temp_local[0][1]).any():
        # 	print(w_temp_local[0][1])
        # 	print('nan found in weights 1')
        # 	raise
        # if np.isnan(w_temp_local[0][2]).any():
        # 	print(w_temp_local[0][2])
        # 	print('nan found in weights 2')
        # 	raise
        # if np.isnan(w_temp_local[0][3]).any():
        # 	print(w_temp_local[0][3])
        # 	print('nan found in weights 3')
        # 	raise
        # if np.isnan(w_temp_local[0][4]).any():
        # 	print(w_temp_local[0][4])
        # 	print('nan found in weights 4')
        # 	raise
        # if np.isnan(w_temp_local[0][5]).any():
        # 	print(w_temp_local[0][5])
        # 	print('nan found in weights 5')
        # 	raise
        for s in range(len(w_temp_local[0])):
            if i == 0:
                w_temp.append(w_temp_local[0][s])
            else:
                w_temp[s] = np.concatenate((w_temp[s], w_temp_local[0][s]), axis=0)
    """"
	  infomration[k] = inn.calc_information_for_epoch(k, interval_information_display, ws_t, params['bins'],
										params['unique_inverse_x'],
										params['unique_inverse_y'],
										params['label'], estimted_labels,
										params['b'], params['b1'], params['len_unique_a'],
										params['pys'], py_hats_temp, params['pxs'], params['py_x'],
										params['pys1'])

	"""
    return w_temp


def get_accuracy(batch_points_test, data_sets, model, sess, j, acc_train_array):
    """Calc the test acc and print the train and test accuracy"""
    acc_array = []
    for i in range(0, len(batch_points_test) - 1):
        batch_xs = data_sets.test.data[batch_points_test[i] : batch_points_test[i + 1]]
        batch_ys = data_sets.test.labels[
            batch_points_test[i] : batch_points_test[i + 1]
        ]
        feed_dict_temp = {model.x: batch_xs, model.labels: batch_ys}
        acc = sess.run([model.accuracy], feed_dict=feed_dict_temp)
        acc_array.append(acc)

    train_acc = np.mean(np.array(acc_train_array))
    test_acc = np.mean(np.array(acc_array))

    # print('Epoch {0} - Test Accuracy: {1:.3f} Train Accuracy: {2:.3f}'.format(j, test_acc, train_acc))

    return train_acc, test_acc


def train_network(
    layerSize,
    num_of_epochs,
    learning_rate_local,
    batch_size,
    indexes,
    save_grads,
    data_sets_org,
    model_type,
    percent_of_train,
    interval_accuracy_display,
    name,
    covn_net,
):
    """Train the nework"""
    tf.compat.v1.reset_default_graph()
    data_sets = data_shuffle(data_sets_org, percent_of_train)
    ws, estimted_label, gradients, infomration, models, weights = [
        [None] * len(indexes) for _ in range(6)
    ]
    loss_func_test, loss_func_train, test_prediction, train_prediction = [
        np.zeros((len(indexes))) for _ in range(4)
    ]
    input_size = data_sets_org.data.shape[1]
    num_of_classes = data_sets_org.labels.shape[1]
    batch_size = np.min([batch_size, data_sets.train.data.shape[0]])
    batch_points = np.rint(
        np.arange(0, data_sets.train.data.shape[0] + 1, batch_size)
    ).astype(dtype=np.int32)
    batch_points_test = np.rint(
        np.arange(0, data_sets.test.data.shape[0] + 1, batch_size)
    ).astype(dtype=np.int32)
    batch_points_all = np.rint(
        np.arange(0, data_sets_org.data.shape[0] + 1, batch_size)
    ).astype(dtype=np.int32)
    if data_sets_org.data.shape[0] not in batch_points_all:
        batch_points_all = np.append(batch_points_all, [data_sets_org.data.shape[0]])
    if data_sets.train.data.shape[0] not in batch_points:
        batch_points = np.append(batch_points, [data_sets.train.data.shape[0]])
    if data_sets.test.data.shape[0] not in batch_points_test:
        batch_points_test = np.append(batch_points_test, [data_sets.test.data.shape[0]])
    # Build the network
    model = build_model(
        model_type,
        layerSize,
        input_size,
        num_of_classes,
        learning_rate_local,
        name,
        covn_net,
    )
    optimizer = model.optimize
    saver = tf.compat.v1.train.Saver(max_to_keep=0)
    init = tf.compat.v1.global_variables_initializer()
    grads = tf.gradients(ys=model.cross_entropy, xs=tf.compat.v1.trainable_variables())
    # Train the network
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        # Go over the epochs
        k = 0
        for j in range(0, num_of_epochs):  # epoch iterations
            epochs_grads = []
            if j in indexes:
                ws[k] = exctract_activity(sess, batch_points_all, model, data_sets_org)
            # Print accuracy
            if (
                np.mod(j, interval_accuracy_display) == 1
                or interval_accuracy_display == 1
            ):
                train_acc, test_acc = get_accuracy(
                    batch_points_test, data_sets, model, sess, j, acc_train_array
                )
                print(
                    "Epoch {0} - Test Accuracy: {1:.3f} Train Accuracy: {2:.3f}".format(
                        j, test_acc, train_acc
                    )
                )
            # Go over the batch_points
            acc_train_array = []
            current_weights = [[] for _ in range(len(model.weights_all))]
            for i in range(0, len(batch_points) - 1):  # train with batches
                train_batch_xs = data_sets.train.data[
                    batch_points[i] : batch_points[i + 1]
                ]
                train_batch_ys = data_sets.train.labels[
                    batch_points[i] : batch_points[i + 1]
                ]
                feed_dict = {model.x: train_batch_xs, model.labels: train_batch_ys}
                _, tr_err = sess.run([optimizer, model.accuracy], feed_dict=feed_dict)
                acc_train_array.append(tr_err)
                if j in indexes:  # logging steps?
                    epochs_grads_temp, loss_tr, weights_local = sess.run(
                        [grads, model.cross_entropy, model.weights_all],
                        feed_dict=feed_dict,
                    )
                    # for grad in epochs_grads_temp:
                    # 	print(grad.shape)
                    # raise
                    epochs_grads.append(epochs_grads_temp)
                    for ii in range(len(current_weights)):
                        current_weights[ii].append(weights_local[ii])

            if j in indexes:
                train_acc, test_acc = get_accuracy(
                    batch_points_test, data_sets, model, sess, j, acc_train_array
                )
                train_prediction[k] = train_acc
                test_prediction[k] = test_acc
                if save_grads:
                    gradients[k] = epochs_grads
                    current_weights_mean = []
                    for ii in range(len(current_weights)):
                        current_weights_mean.append(
                            np.mean(np.array(current_weights[ii]), axis=0)
                        )
                    weights[k] = current_weights_mean
                # Save the model
                # write_meta = True if k == 0 else False
                # saver.save(sess, model.save_file, global_step=k, write_meta_graph=write_meta)
                k += 1

    network = {}
    network["ws"] = ws
    network["weights"] = weights
    network["test_prediction"] = test_prediction
    network["train_prediction"] = train_prediction
    network["loss_test"] = loss_func_test
    network["loss_train"] = loss_func_train
    network["gradients"] = gradients
    network["model"] = model
    return network


def train_network_pytorch(
    layerSize,
    num_of_epochs,
    learning_rate_local,
    batch_size,
    indexes,
    save_grads,
    data_sets_org,
    model_type,
    percent_of_train,
    interval_accuracy_display,
    name,
    covn_net,
    is_mnist,
):
    # initialize data
    ws, estimted_label, gradients, information, models, weights = [
        [None] * len(indexes) for _ in range(6)
    ]
    loss_func_test, loss_func_train, test_prediction, train_prediction = [
        np.zeros((len(indexes))) for _ in range(4)
    ]

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataset
    (
        train_dataloader,
        test_dataloader,
        all_data_dataloader,
        input_size,
        num_of_classes,
    ) = data_shuffle_pytorch(
        data_sets_org, percent_of_train, batch_size, is_mnist=is_mnist
    )

    # build model
    model = mo.Model(
        model_type, layerSize, input_size, num_of_classes, name, covn_net
    ).to(device)

    accuracy_metric = Accuracy(task="multiclass", num_classes=num_of_classes).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()  # Softmax implemented internally
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate_local, momentum=0.95
    )

    # train model
    k = 0
    for epoch in range(num_of_epochs):
        if epoch in indexes:
            # get the activation of all layers
            ws[k] = extract_activity_pytorch(all_data_dataloader, model, device)

        # Print accuracy
        if (
            np.mod(epoch, interval_accuracy_display) == 1
            or interval_accuracy_display == 1
        ):
            train_acc, test_acc = get_accuracy_pytorch(
                train_dataloader, test_dataloader, model, accuracy_metric, device
            )
            print(
                "Epoch {0} - Test Accuracy: {1:.3f} Train Accuracy: {2:.3f}".format(
                    epoch, test_acc, train_acc
                )
            )

        #  Train all batches
        for data_idx, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs, _ = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch in indexes:
            train_acc, test_acc = get_accuracy_pytorch(
                train_dataloader, test_dataloader, model, accuracy_metric, device
            )
            train_prediction[k] = train_acc
            test_prediction[k] = test_acc
            if save_grads:
                pass

            k += 1

    model.to("cpu")
    network = {}
    network["ws"] = ws
    network["weights"] = weights
    network["test_prediction"] = test_prediction
    network["train_prediction"] = train_prediction
    network["loss_test"] = loss_func_test
    network["loss_train"] = loss_func_train
    network["gradients"] = gradients
    network["model"] = model
    return network


def extract_activity_pytorch(dataloader, model, device):
    model.to(device)

    w_temp = []
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        _, layers = model(inputs)
        for layer in layers:
            layer.retain_grad()

        for layer_index in range(len(layers)):
            layer = layers[layer_index].detach().cpu().numpy()
            if i == 0:
                w_temp.append(layer)
            else:
                w_temp[layer_index] = np.concatenate(
                    (w_temp[layer_index], layer), axis=0
                )
    return w_temp


def get_accuracy_pytorch(
    train_dataloader, test_dataloader, model, accuracy_metric, device
):
    model.to(device)

    train_acc = []
    for data in train_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, _ = model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1)
        labels = torch.nn.functional.softmax(labels, dim=1).argmax(dim=1)

        accuracy_metric.update(outputs, labels)
        train_acc.append(accuracy_metric.compute().detach().cpu().numpy())
        accuracy_metric.reset()
    train_acc = np.mean(train_acc)

    test_acc = []
    for data in test_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, _ = model(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1)
        labels = torch.nn.functional.softmax(labels, dim=1).argmax(dim=1)

        accuracy_metric.update(outputs, labels)
        test_acc.append(accuracy_metric.compute().detach().cpu().numpy())
        accuracy_metric.reset()
    test_acc = np.mean(test_acc)

    return train_acc, test_acc
