"""
Train % plot networks in the information plane
"""
from idnns.networks import information_network as inet
def main():
    # #Build the network
    # print ('Building the network')
    net = inet.informationNetwork()
    # net.dir_saved = 'jobs/net_sampleLen=1_nDistSmpls=1_layerSizes=10,7,5,4,3_nEpoch=8000_batch=512_nRepeats=1_nEpochInds=274_LastEpochsInds=7999_DataName=var_u_lr=0.0004/'
    net.print_information()
    print ('Start running the network')
    net.run_network()
    print ('Saving data')
    net.save_data()
    print ('Ploting figures')
    #Plot the newtork
    net.plot_network()
if __name__ == '__main__':
    main()

