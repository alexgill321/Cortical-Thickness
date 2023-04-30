from tensorflow import keras
from abc import abstractmethod


class AEModel(keras.Model):
    def __init__(self, encoder, decoder):
        super(AEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    @abstractmethod
    def call(self, inputs, training=False, mask=None):
        pass

    @abstractmethod
    def train_step(self, batch_data):
        pass

    @abstractmethod
    def test_step(self, batch_data):
        pass

    @abstractmethod
    def get_config(self):
        pass

    @classmethod
    def from_config(cls, config, custom_objects=None):
        pass


