import os
import pandas as pd
from math import floor


def import_data(test_ratio):
    data = pd.read_csv('https://query.data.world/s/culciexydc2njqbyaqxayl7rleyhwf')
    data = data.sample(frac=1).reset_index(drop=True)
    data.to_csv("data.csv", index=False)
    num_test = int(test_ratio*data.shape[0])
    testset = data[:num_test]
    trainset = data[num_test:]
    return trainset, testset


def splitset(dataset, parts):
    n = dataset.shape[0]
    local_n = floor(n/parts)
    result = []
    for i in range(parts):
        result.append(dataset[i*local_n: (i+1)*local_n])
    return result


if __name__ == '__main__':

    nr_of_datasets = 10

    trainset, testset = import_data(0.1)
    trainsets = splitset(trainset, nr_of_datasets)
    testsets = splitset(testset, nr_of_datasets)

    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/{}clients'.format(nr_of_datasets)):
        os.mkdir('data/{}clients'.format(nr_of_datasets))

    for i in range(nr_of_datasets):
        if not os.path.exists('data/{}clients/client'.format(nr_of_datasets) + str(i)):
            os.mkdir('data/{}clients/client'.format(nr_of_datasets) + str(i))
        trainsets[i].to_csv('data/{}clients/client'.format(nr_of_datasets) + str(i)+'/train.csv', index=False)
        testsets[i].to_csv('data/{}clients/client'.format(nr_of_datasets) + str(i) + '/test.csv', index=False)