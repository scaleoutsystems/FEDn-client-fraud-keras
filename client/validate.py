import sys
from data.read_data import read_data
import json
import yaml


def validate(model):
    print("-- RUNNING VALIDATION --", flush=True)
    try:
        train, train_valx, train_valy = read_data('../data/train.csv')
        test, test_valx, test_valy = read_data('../data/test.csv')
        trainScores = model.evaluate(train, train, verbose=0)
        print('Training loss:', trainScores[0])
        print('Training accuracy:', trainScores[1])
        testScores = model.evaluate(test, test, verbose=0)
        print('Test loss:', testScores[0])
        print('Test accuracy:', testScores[1])

    except Exception as e:
        print("failed to validate the model {}".format(e), flush=True)
        raise
    
    report = { 
                "classification_report": 'unevaluated',
                "training_loss": trainScores[0],
                "training_accuracy": trainScores[1],
                "test_loss": testScores[0],
                "test_accuracy": testScores[1],
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report


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

    report = validate(model)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))

