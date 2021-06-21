# FEDn-client-fraud-keras

This repository contains a FEDn client implementation for federated training of a Keras autoencoder model for 
credit card fraud detection using the public dataset https://www.kaggle.com/mlg-ulb/creditcardfraud.  

In this example the credit card transaction dataset is divided (IID) into a configurable number of clients. 
In each round update, the client trains the autoencoder model locally for one epoch using a batch size of 32 (Adam) 
optimizer.

To download and prepare the partitioned dataset:

    $ python create_data_partitions.py 
    
This downloads the full dataset into the file 'data.csv', and creates random partitions in data/clients/.

