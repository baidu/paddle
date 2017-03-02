import random

__all__ = ['BaseDataSet']


class BaseDataSet(object):
    def __init__(self, random_seed):
        self.__random__ = random.Random()
        self.__random__.seed(random_seed)

    def train_data(self):
        raise NotImplemented()

    def test_data(self):
        raise NotImplemented()
