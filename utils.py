class Bunch(dict):
    def __init__(self, dict):
        self.update(dict)
        self.__dict__ = self