# FEDn-client-fraud-keras

This repository contains a FEDn client implementation for federated training of a Keras autoencoder model for 
credit card fraud detection using the public dataset https://www.kaggle.com/mlg-ulb/creditcardfraud.  

In this example the credit card transaction dataset is divided (IID) into a configurable number of clients. 
In each round update, the client trains the autoencoder model locally for one epoch using a batch size of 32 (Adam) 
optimizer.

To download and prepare the partitioned dataset:

    $ python create_data_partitions.py 
    
This downloads the full dataset into the file 'data.csv', and creates random partitions in data/clients/.

## Start a client 

Clone this repository and then download the client.yaml config file from the FEDn UI (Network view) and copy it into the main repostitory folder. 

To build the client environment: 
    
    $ docker build . -t fraud-client:latest
    
To start a client (using Docker): 

     $ docker run -it -v $(pwd)/data/clients/0:/app/data -v $(pwd)/client.yaml:/app/client.yaml fraud-client:latest fedn run client -in client.yaml
