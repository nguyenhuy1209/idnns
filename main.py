"""
Train % plot networks in the information plane
"""

from idnns.networks import information_network as inet


def main():
    # Build the network
    print("Building the network")
    args = {
        "d_name": "MNIST",
        "save_ws": True,
        "calc_information": True,
        "calc_information_last": True,
        "save_grads": True,
        "num_epochs": 10000,
        "num_of_repeats": 1,
        "num_of_samples": 10000,
        "activation_function": 7,
        "batch_size": 300,
        "learning_rate": 0.004,
        "net_type": "[512,256,128,64]",  # Fig 3
    }
    net = inet.informationNetwork(args_additional=args)
    net.print_information()
    print("Start running the network")
    net.run_network()
    print("Saving data")
    net.save_data()
    print("Ploting figures")
    # net.dir_saved = "/Users/huy/Code/idnns/jobs/net_sampleLen=1_nDistSmpls=1_layerSizes=10,8,6,4_nEpoch=10000_batch=4096_nRepeats=5_nEpochInds=3496_LastEpochsInds=9998_DataName=var_u_lr=0.004_renyi_abs"
    net.plot_network()


if __name__ == "__main__":
    main()
