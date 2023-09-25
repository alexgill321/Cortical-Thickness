""" Unit tests for the AAE model and its components.

This script contains unit tests for the AAE model and its components. The tests are run using the unittest framework.
The tests are run using the following command:
    python -m unittest UnitTests.aae_unit_tests

The tests are run on the following functions:
    - AAEEncoder
    - AAEDecoder
    - AAEDiscriminator
    - AAE
    - discriminator_loss
    - AAEOptimizer
    - create_aae
    - create_aae_scheduler
"""

import unittest
import numpy as np
import tensorflow as tf
from models.aae_models import AAE, AAEDiscriminator, AAEDecoder, AAEEncoder, discriminator_loss, AAEOptimizer
import os
from utils import data_validation
from modelUtils.lr_utils import MultiOptimizerLearningRateScheduler, CyclicLR, ExponentialDecayScheduler
from keras import backend as k


class TestAAEModel(unittest.TestCase):
    """ Unit tests for the AAE model and its components.

    This class contains unit tests for the AAE model and its components, particularly unit tests for specific model
    components and unit tests for the model as a whole. The tests are run using the unittest framework.
    """
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

        self.encoder = AAEEncoder(self.h_dim, self.z_dim)
        self.decoder = AAEDecoder(self.input_dim, self.h_dim)
        self.discriminator = AAEDiscriminator(self.z_dim, self.h_dim)
        self.aae = AAE(self.encoder, self.decoder, self.discriminator, self.z_dim)

    def tearDown(self):
        del self.encoder
        del self.decoder
        del self.discriminator
        del self.aae
        del self.train_data
        del self.val_data
        del self.test_data

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

    def test_default_compilation(self):
        self.aae.compile()
        self.assertEqual(self.aae.optimizer.autoencoder_optimizer.get_config()['name'], 'Adam')
        self.assertEqual(self.aae.optimizer.generator_optimizer.get_config()['name'], 'Adam')
        self.assertEqual(self.aae.optimizer.discriminator_optimizer.get_config()['name'], 'Adam')
        self.assertEqual(self.aae.optimizer.name, 'AAEOptimizer')
        self.assertIsInstance(self.aae.generator_loss_fn, tf.keras.losses.BinaryCrossentropy)
        self.assertEqual(self.aae.discriminator_loss_fn, discriminator_loss)
        self.assertIsInstance(self.aae.autoencoder_loss_fn, tf.keras.losses.MeanSquaredError)

    def test_change_opt(self):
        enc_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        gen_optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        opt = AAEOptimizer(enc_optimizer, gen_optimizer, disc_optimizer)

        self.aae.compile(optimizer=opt)
        self.assertEqual(self.aae.optimizer.autoencoder_optimizer.get_config()['name'], 'SGD')
        self.assertEqual(self.aae.optimizer.generator_optimizer.get_config()['name'], 'Adagrad')
        self.assertEqual(self.aae.optimizer.discriminator_optimizer.get_config()['name'], 'RMSprop')
        self.assertEqual(self.aae.optimizer.name, 'AAEOptimizer')

    def test_gradients_applied(self):
        # Create a simple dataset

        train_data = self.train_data.batch(self.batch_size)
        train_batches = iter(train_data)
        self.aae.compile()

        # Save the initial weights
        initial_encoder_weights = self.aae.encoder.get_weights()
        initial_decoder_weights = self.aae.decoder.get_weights()
        initial_discriminator_weights = self.aae.discriminator.get_weights()

        # Train the model for a few steps
        num_steps = 5
        for _ in range(num_steps):
            data = next(train_batches)
            self.aae.train_on_batch(data[0], data[1])

        # Check if the weights have been updated
        updated_encoder_weights = self.aae.encoder.get_weights()
        updated_decoder_weights = self.aae.decoder.get_weights()
        updated_discriminator_weights = self.aae.discriminator.get_weights()

        for initial_weight, updated_weight in zip(initial_encoder_weights, updated_encoder_weights):
            self.assertFalse(np.allclose(initial_weight, updated_weight), "Encoder weights were not updated")

        for initial_weight, updated_weight in zip(initial_decoder_weights, updated_decoder_weights):
            self.assertFalse(np.allclose(initial_weight, updated_weight), "Decoder weights were not updated")

        for initial_weight, updated_weight in zip(initial_discriminator_weights, updated_discriminator_weights):
            self.assertFalse(np.allclose(initial_weight, updated_weight), "Discriminator weights were not updated")


class TestAAELearningRateScheduler(unittest.TestCase):
    """ Test the learning rate scheduler for the AAE model

    These test cases check if the learning rate scheduler is correctly applied to the optimizers of the AAE model.
    """
    def setUp(self):
        k.clear_session()
        tf.compat.v1.reset_default_graph()
        self.h_dim = [100, 100]
        self.z_dim = 20
        self.batch_size = 256
        self.epochs = 50

        cur = os.getcwd()
        filepath = os.path.join(cur, '../outputs/megasample_cleaned.csv')
        self.train_data, self.val_data, self.test_data = data_validation(filepath)

        self.input_dim = self.train_data.element_spec[0].shape[0]
        # Instantiate models
        self.encoder = AAEEncoder(self.h_dim, self.z_dim)
        self.decoder = AAEDecoder(self.input_dim, self.h_dim)
        self.discriminator = AAEDiscriminator(self.z_dim, self.h_dim)
        self.opt = AAEOptimizer()
        self.aae = AAE(self.encoder, self.decoder, self.discriminator, self.z_dim)

    def tearDown(self):
        del self.aae
        del self.encoder
        del self.decoder
        del self.discriminator
        del self.train_data
        del self.val_data
        del self.test_data
        del self.opt

    def test_multi_opt_lr_scheduler_with_aae_optimizer(self):
        base_lr = 0.001
        max_lr = 0.006
        step_size = 20
        decay_rate = 0.9

        autoencoder_scheduler = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="exp_range")
        generator_scheduler = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="triangular2")
        discriminator_scheduler = ExponentialDecayScheduler(initial_lr=base_lr, decay_rate=decay_rate,
                                                            decay_steps=step_size)
        self.aae.compile(optimizer=self.opt)
        optimizers = [self.aae.optimizer.autoencoder_optimizer, self.aae.optimizer.generator_optimizer,
                      self.aae.optimizer.discriminator_optimizer]
        schedulers = [autoencoder_scheduler, generator_scheduler, discriminator_scheduler]
        lr_and_opt = zip(schedulers, optimizers)
        scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
        train_data = self.train_data.batch(self.batch_size)
        history = self.aae.fit(train_data, epochs=self.epochs, callbacks=[scheduler], verbose=0)

        # Check if the learning rates from the logs are within the expected range
        tolerance = 1e-6
        for lr in history.history['ae_lr']:
            self.assertGreaterEqual(lr, base_lr)
            self.assertLessEqual(lr, max_lr+tolerance)
        for lr in history.history['gen_lr']:
            self.assertGreaterEqual(lr, base_lr)
            self.assertLessEqual(lr, max_lr+tolerance)
        for lr in history.history['dc_lr']:
            self.assertGreaterEqual(lr, base_lr * (decay_rate**self.epochs/step_size))
            self.assertLessEqual(lr, base_lr+tolerance)

    def test_aae_opt_with_gen_lr_scheduler(self):
        base_lr = 0.001
        max_lr = 0.006
        step_size = 20

        generator_scheduler = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="triangular2")

        self.aae.compile(optimizer=self.opt)
        optimizers = [self.aae.optimizer.generator_optimizer]
        schedulers = [generator_scheduler]
        lr_and_opt = zip(schedulers, optimizers)
        scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
        train_data = self.train_data.batch(self.batch_size)
        history = self.aae.fit(train_data, epochs=self.epochs, callbacks=[scheduler], verbose=0)

        # Check if the learning rates from the logs are within the expected range
        tolerance = 1e-6
        for lr in history.history['ae_lr']:
            self.assertAlmostEqual(lr, 0.001)
        for lr in history.history['gen_lr']:
            self.assertGreaterEqual(lr, base_lr)
            self.assertLessEqual(lr, max_lr+tolerance)
        for lr in history.history['dc_lr']:
            self.assertAlmostEqual(lr, 0.001)

    def test_aae_opt_with_dc_lr_scheduler(self):
        base_lr = 0.001
        decay_rate = 0.9
        step_size = 20

        discriminator_scheduler = ExponentialDecayScheduler(initial_lr=base_lr, decay_rate=decay_rate,
                                                            decay_steps=step_size)

        self.aae.compile(optimizer=self.opt)
        optimizers = [self.aae.optimizer.discriminator_optimizer]
        schedulers = [discriminator_scheduler]
        lr_and_opt = zip(schedulers, optimizers)
        scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
        train_data = self.train_data.batch(self.batch_size)
        history = self.aae.fit(train_data, epochs=self.epochs, callbacks=[scheduler], verbose=0)

        # Check if the learning rates from the logs are within the expected range
        tolerance = 1e-6
        for lr in history.history['ae_lr']:
            self.assertAlmostEqual(lr, 0.001)
        for lr in history.history['gen_lr']:
            self.assertAlmostEqual(lr, 0.001)
        for lr in history.history['dc_lr']:
            self.assertGreaterEqual(lr, base_lr * (decay_rate**self.epochs/step_size))
            self.assertLessEqual(lr, base_lr+tolerance)

    def test_aae_opt_with_ae_lr_scheduler(self):
        base_lr = 0.001
        max_lr = 0.006
        step_size = 20

        autoencoder_scheduler = CyclicLR(base_lr=base_lr, step_size=step_size, max_lr=max_lr, mode="exp_range")

        self.aae.compile(optimizer=self.opt)
        optimizers = [self.aae.optimizer.autoencoder_optimizer]
        schedulers = [autoencoder_scheduler]
        lr_and_opt = zip(schedulers, optimizers)
        scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
        train_data = self.train_data.batch(self.batch_size)
        history = self.aae.fit(train_data, epochs=self.epochs, callbacks=[scheduler], verbose=0)

        # Check if the learning rates from the logs are within the expected range
        tolerance = 1e-6
        for lr in history.history['ae_lr']:
            self.assertGreaterEqual(lr, base_lr)
            self.assertLessEqual(lr, max_lr+tolerance)
        for lr in history.history['gen_lr']:
            self.assertAlmostEqual(lr, 0.001)
        for lr in history.history['dc_lr']:
            self.assertAlmostEqual(lr, 0.001)


if __name__ == '__main__':
    unittest.main()

#%%
