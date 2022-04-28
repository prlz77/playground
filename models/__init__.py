from models.classifier import Classifier

def get_model(hparams):
    if hparams['model'] == "classifier":
        return Classifier(**hparams)