"""
Train % plot networks in the information plane
"""

from idnns.networks import information_network as inet


def main():
    # Build the network
    print("Building the network")
    args = {
        "data_name": "MNIST",
        "save_ws": True,
        "calc_information": True,
        "calc_information_last": True,
        "save_grads": True,
        "num_epochs": 1001,
        "num_of_repeats": 1,
        "num_of_samples": 100,
        "activation_function": 0,
        "batch_size": 1024,
        "learning_rate": 0.002,
        "net_type": "[512, 256, 128, 64]",  # Fig 3
        "num_samples": 50,
    }
    net = inet.informationNetwork(args_additional=args)
    net.print_information()
    print("Start running the network")
    net.run_network()
    print("Saving data")
    net.save_data()
    print("Ploting figures")
    net.plot_network()


if __name__ == "__main__":
    main()
