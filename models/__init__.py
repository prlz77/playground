from models.classifier import Classifier
from models.double_classifier import DoubleClassifier

def get_model(hparams):
    if hparams['model'] == "classifier":
        return Classifier(**hparams)
    elif hparams['model'] == "double_classifier":
        return DoubleClassifier(**hparams)
    else:
        raise ValueError