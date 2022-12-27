import json
class Bunch(dict):
    def __init__(self, dict):
        self.update(dict)
        self.__dict__ = self

def load_json(path):
    with open(path, 'r') as infile:
        return json.load(infile)