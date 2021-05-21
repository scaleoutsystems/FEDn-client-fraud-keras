# FEDn-client-fraud-keras

This repository contains a FEDn client and seed model for federated training of a Keras autoencoder model for 
credit card fraud testing using a public dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud.  

The credit card transaction dataset is here evenly divided (IID) into 10 clients (but this can be easily modified) by executing "load_datasets.py". 
The clients train the autoencoder model locally one epoch in each round with a batch size of 32, using the Adam 
optimizer.

