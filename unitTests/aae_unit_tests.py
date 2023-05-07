import unittest
import numpy as np
import tensorflow as tf
from models.aae_models import AAE, AAEDiscriminator, AAEDecoder, AAEEncoder
import os
from utils import data_validation


class TestAAEModel(unittest.TestCase):
    def setUp(self):
        # Model parameters
        self.h_dim = [100, 100]
        self.z_dim = 20
        self.batch_size = 256
        self.epochs = 200

        cur = os.getcwd()
        filepath = os.path.join(cur, '../outputs/megasample_cleaned.csv')
        self.train_data, self.val_data, self.test_data = data_validation(filepath)

        self.input_dim = self.train_data.element_spec[0].shape[0]
        # Instantiate models
        self.encoder = AAEEncoder(self.input_dim, self.h_dim, self.z_dim)
        self.decoder = AAEDecoder(self.z_dim, self.input_dim, self.h_dim)
        self.discriminator = AAEDiscriminator(self.z_dim, self.h_dim)

        self.aae = AAE(self.encoder, self.decoder, self.discriminator, self.z_dim)

    def test_encoder_output_shape(self):
        inputs = tf.random.normal([32, self.input_dim])
        outputs = self.encoder(inputs)
        self.assertEqual(outputs.shape, (32, self.z_dim))

    def test_decoder_output_shape(self):
        inputs = tf.random.normal([32, self.z_dim])
        outputs = self.decoder(inputs)
        self.assertEqual(outputs.shape, (32, self.input_dim))

    def test_discriminator_output_shape(self):
        inputs = tf.random.normal([32, self.z_dim])
        outputs = self.discriminator(inputs)
        self.assertEqual(outputs.shape, (32, 1))

    def test_aae_output_shape(self):
        self.aae.compile()
        inputs = next(iter(self.train_data.batch(self.batch_size)))
        outputs = self.aae(inputs)
        self.assertEqual(outputs.shape, (self.batch_size, self.input_dim))

    def test_autoencoder_loss(self):
        self.aae.compile()
        batch_size = 10
        inputs = tf.random.normal([batch_size, self.input_dim])

        # Expected MSE loss
        encoder_output = self.encoder(inputs)
        decoder_output = self.decoder(encoder_output)
        expected_loss = np.mean(np.square(inputs.numpy() - decoder_output.numpy()))

        # Calculate loss using the model's loss function
        ae_loss = self.aae.autoencoder_loss_fn(inputs, decoder_output)
        loss_value = ae_loss.numpy()

        self.assertAlmostEqual(loss_value, expected_loss, places=5)

    def test_discriminator_loss(self):
        self.aae.compile()
        batch_size = 10
        real_distribution = tf.random.normal([batch_size, self.z_dim], mean=0.0, stddev=1.0)
        encoder_output = self.encoder(tf.random.normal([batch_size, self.input_dim]))

        # Calculate logits
        dc_real = self.discriminator(real_distribution)
        dc_fake = self.discriminator(encoder_output)

        # Expected Binary Crossentropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        expected_loss_real = cross_entropy(tf.ones_like(dc_real), dc_real)
        expected_loss_fake = cross_entropy(tf.zeros_like(dc_fake), dc_fake)
        expected_loss = (expected_loss_real + expected_loss_fake).numpy()

        # Calculate loss using the model's loss function
        dc_loss = self.aae.discriminator_loss_fn(dc_real, dc_fake)
        loss_value = dc_loss.numpy()

        self.assertAlmostEqual(loss_value, expected_loss, places=5)

    def test_generator_loss(self):
        self.aae.compile()
        batch_size = 10
        encoder_output = self.encoder(tf.random.normal([batch_size, self.input_dim]))

        # Calculate logits
        dc_fake = self.discriminator(encoder_output)

        # Expected Binary Crossentropy loss
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        expected_loss = cross_entropy(tf.ones_like(dc_fake), dc_fake).numpy()

        # Calculate loss using the model's loss function
        dc_loss = self.aae.generator_loss_fn(tf.ones_like(dc_fake), dc_fake)
        loss_value = dc_loss.numpy()

        self.assertAlmostEqual(loss_value, expected_loss, places=5)


if __name__ == '__main__':
    unittest.main()

#%%
