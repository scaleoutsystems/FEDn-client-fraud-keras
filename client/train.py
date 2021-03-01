from __future__ import print_function
import sys
import yaml
from data.read_data import read_data


def train(model, filename, settings):
    print("-- RUNNING TRAINING --", flush=True)
    train_x, _, _ = read_data(filename)
    model.fit(train_x, train_x, batch_size=settings['batch_size'],
              epochs=settings['epochs'])
    return model


if __name__ == '__main__':
    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.kerasweights import KerasWeightsHelper
    from models.autocoder import create_seed_model

    helper = KerasWeightsHelper()
    weights = helper.load_model(sys.argv[1])

    model = create_seed_model()
    model.set_weights(weights)

    model = train(model, '../data/train.csv', settings)
    helper.save_model(model.get_weights(), sys.argv[2])
