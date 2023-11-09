""" Unit tests for the variational autoencoder.

This script contains unit test for the VAE model and its components. The tests are designed to test the VAE model in
isolation. The main purpose of these tests is to verify the desired functionality of the VAE model, and the encoder and
decoder components. Also tested is the validation and training of the VAE model via the utility functions in
vae_utils.py.

"""

import unittest
import numpy as np
import tensorflow as tf
from models.vae_models import VAE, calc_kl_loss, create_vae_encoder, create_vae_decoder, r2_feat_score, r2_score
import os
from utils import generate_data_thickness_only, generate_feature_names
from modelUtils.lr_utils import MultiOptimizerLearningRateScheduler, CyclicLR, ExponentialDecayScheduler
from modelUtils.vae_utils import train_val_vae, create_vae, VAECrossValidator, save_vae, load_vae, \
    get_filename_from_params, create_param_grid, load_or_train_model
import vis_utils as vu
import shutil
from vaeModelAnalyzer import VAEModelAnalyzer
import matplotlib.pyplot as plt


class TestVAEModel(unittest.TestCase):
    def setUp(self):
        cur = os.getcwd()
        filepath = os.path.join(cur, '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
        self.train_data, self.val_data, self.test_data, _ = generate_data_thickness_only(filepath)
        self.h_dim = [100, 100]
        self.z_dim = 20
        self.input_dim = self.train_data.element_spec[0].shape[0]
        self.batch_size = 128
        self.epochs = 10
        self.lr = 0.001
        self.encoder = create_vae_encoder(self.input_dim, self.h_dim, self.z_dim)
        self.decoder = create_vae_decoder(self.z_dim, self.h_dim, self.input_dim)
        self.vae = VAE(self.encoder, self.decoder)

    def test_encoder_output_shape(self):
        x = tf.random.normal(shape=(self.batch_size, self.input_dim))
        z_mean, z_log_var = self.encoder(x)
        self.assertEqual(z_mean.shape, (self.batch_size, self.z_dim))
        self.assertEqual(z_log_var.shape, (self.batch_size, self.z_dim))

    def test_decoder_output_shape(self):
        z = tf.random.normal(shape=(self.batch_size, self.z_dim))
        x = self.decoder(z)
        self.assertEqual(x.shape, (self.batch_size, self.input_dim))

    def test_encoder_layer_shapes_simple(self):
        x = tf.random.normal(shape=(self.batch_size, self.input_dim))
        z_mean, z_log_var = self.encoder(x)
        layers = self.encoder.layers
        self.assertEqual(len(layers), len(self.h_dim) * 2 + 3)
        self.assertEqual(self.encoder.layers[0].input_shape[0], (None, self.input_dim))
        self.assertEqual(self.encoder.layers[-1].output_shape, (None, self.z_dim))
        for i in range(len(self.h_dim)):
            self.assertEqual(self.encoder.layers[i * 2 + 1].output_shape, (None, self.h_dim[i]))

    def test_decoder_layer_shapes_simple(self):
        z = tf.random.normal(shape=(self.batch_size, self.z_dim))
        x = self.decoder(z)
        layers = self.decoder.layers
        self.assertEqual(len(layers), len(self.h_dim) * 2 + 2)
        self.assertEqual(self.decoder.layers[0].input_shape[0], (None, self.z_dim))
        self.assertEqual(self.decoder.layers[-1].output_shape, (None, self.input_dim))
        for i in range(len(self.h_dim)):
            self.assertEqual(self.decoder.layers[i * 2 + 1].output_shape, (None, self.h_dim[i]))

    def test_encoder_layer_shapes_complex(self):
        self.h_dim = [300, 200, 100]
        self.encoder = create_vae_encoder(self.input_dim, self.h_dim, self.z_dim)
        x = tf.random.normal(shape=(self.batch_size, self.input_dim))
        z_mean, z_log_var = self.encoder(x)
        layers = self.encoder.layers
        self.assertEqual(len(layers), len(self.h_dim) * 2 + 3)
        self.assertEqual(self.encoder.layers[0].input_shape[0], (None, self.input_dim))
        self.assertEqual(self.encoder.layers[-1].output_shape, (None, self.z_dim))
        for i in range(len(self.h_dim)):
            self.assertEqual(self.encoder.layers[i * 2 + 1].output_shape, (None, self.h_dim[i]))

    def test_decoder_layer_shapes_complex(self):
        self.h_dim = [300, 200, 100]
        self.decoder = create_vae_decoder(self.z_dim, self.h_dim, self.input_dim)
        z = tf.random.normal(shape=(self.batch_size, self.z_dim))
        x = self.decoder(z)
        layers = self.decoder.layers
        self.assertEqual(len(layers), len(self.h_dim) * 2 + 2)
        self.assertEqual(self.decoder.layers[0].input_shape[0], (None, self.z_dim))
        self.assertEqual(self.decoder.layers[-1].output_shape, (None, self.input_dim))
        self.h_dim = self.h_dim[::-1]
        for i in range(len(self.h_dim)):
            self.assertEqual(self.decoder.layers[i * 2 + 1].output_shape, (None, self.h_dim[i]))

    def test_vae_output_shape(self):
        train_data = self.train_data.batch(self.batch_size)
        train_batches = iter(train_data)
        data = next(train_batches)
        self.vae.compile()
        z_mean, z_log_var, z, x_hat = self.vae(data)
        self.assertEqual(x_hat.shape, (self.batch_size, self.input_dim))
        self.assertEqual(z_mean.shape, (self.batch_size, self.z_dim))
        self.assertEqual(z_log_var.shape, (self.batch_size, self.z_dim))

    def test_vae_reconstruction_loss(self):
        train_data = self.train_data.batch(self.batch_size)
        train_batches = iter(train_data)
        data = next(train_batches)
        self.vae.compile()
        z_mean, z_log_var, z, x_hat = self.vae(data)
        loss = self.vae.reconstruction_loss_fn(data[0], x_hat)
        est_loss = tf.keras.losses.MeanSquaredError()(data[0], x_hat)
        self.assertAlmostEqual(loss, est_loss)

    def test_vae_kl_loss(self):
        train_data = self.train_data.batch(self.batch_size)
        train_batches = iter(train_data)
        self.vae.compile()
        data = next(train_batches)
        z_mean, z_log_var, z, x_hat = self.vae(data)
        loss = self.vae.kl_loss_fn(z_mean, z_log_var)
        est_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        self.assertAlmostEqual(loss.numpy(), est_loss.numpy())

    def test_r2_feat_score(self):
        actual = tf.constant([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]], dtype=tf.float32)
        pred = tf.constant([[1.1, 1.9, 2.5],
                            [4.6, 5.2, 6.3],
                            [6.3, 7.8, 5.5]], dtype=tf.float32)
        # Should come out as [0.952, 0.995, 0.301]
        r2 = r2_feat_score(actual, pred)
        self.assertEqual(r2.shape, (3,))
        r2 = r2.numpy()
        self.assertAlmostEqual(r2[0], 0.952, places=3)
        self.assertAlmostEqual(r2[1], 0.995, places=3)
        self.assertAlmostEqual(r2[2], 0.301, places=3)

    def test_r2_score(self):
        actual = tf.constant([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0],
                              [7.0, 8.0, 9.0]], dtype=tf.float32)
        pred = tf.constant([[1.1, 1.9, 2.5],
                            [4.6, 5.2, 6.3],
                            [6.3, 7.8, 5.5]], dtype=tf.float32)
        # Should come out as [0.865, 0.755, -5.39]
        r2 = r2_score(actual, pred)
        self.assertEqual(r2.shape, (3,))
        self.assertAlmostEqual(r2[0], 0.865, places=3)
        self.assertAlmostEqual(r2[1], 0.755, places=3)
        self.assertAlmostEqual(r2[2], -5.39, places=3)

    def test_default_compilation(self):
        self.vae.compile()
        self.assertIsInstance(self.vae.optimizer, tf.keras.optimizers.Adam)
        self.assertIsInstance(self.vae.reconstruction_loss_fn, tf.keras.losses.MeanSquaredError)
        self.assertEqual(self.vae.kl_loss_fn, calc_kl_loss)

    def test_custom_compilation(self):
        self.vae.compile(optimizer=tf.keras.optimizers.RMSprop(),
                         reconstruction_loss_fn=tf.keras.losses.MeanAbsoluteError(),
                         kl_loss_fn=tf.keras.losses.MeanSquaredError())
        self.assertIsInstance(self.vae.optimizer, tf.keras.optimizers.RMSprop)
        self.assertIsInstance(self.vae.reconstruction_loss_fn, tf.keras.losses.MeanAbsoluteError)
        self.assertIsInstance(self.vae.kl_loss_fn, tf.keras.losses.MeanSquaredError)

    def test_gradients_applied(self):
        # Create a simple dataset
        train_data = self.train_data.batch(self.batch_size)
        train_batches = iter(train_data)
        self.vae.compile()

        # Save the initial weights
        initial_encoder_weights = self.vae.encoder.get_weights()
        initial_decoder_weights = self.vae.decoder.get_weights()

        # Train the model for a few steps
        num_steps = 5
        for _ in range(num_steps):
            data = next(train_batches)
            self.vae.train_on_batch(data[0], data[1])

        # Check if the weights have been updated
        updated_encoder_weights = self.vae.encoder.get_weights()
        updated_decoder_weights = self.vae.decoder.get_weights()

        for initial_weight, updated_weight in zip(initial_encoder_weights, updated_encoder_weights):
            self.assertFalse(np.allclose(initial_weight, updated_weight), "Encoder weights were not updated")

        for initial_weight, updated_weight in zip(initial_decoder_weights, updated_decoder_weights):
            self.assertFalse(np.allclose(initial_weight, updated_weight), "Decoder weights were not updated")


class TestVAELearningRateScheduler(unittest.TestCase):
    def setUp(self):
        cur = os.getcwd()
        filepath = os.path.join(cur, '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
        self.train_data, self.val_data, self.test_data = generate_data_thickness_only(filepath)
        self.input_dim = self.train_data.element_spec[0].shape[0]
        self.batch_size = 128
        self.epochs = 10
        self.encoder = create_vae_encoder(self.input_dim, [100, 100], 20)
        self.decoder = create_vae_decoder(20, [100, 100], self.input_dim)
        self.vae = VAE(self.encoder, self.decoder)
        self.base_lr = 0.001
        self.max_lr = 0.006
        self.step_size = 20
        self.decay_rate = 0.9

    def test_multi_opt_lr_scheduler_with_cyclic(self):
        self.vae.compile()
        opt_scheduler = [CyclicLR(self.base_lr, self.step_size, self.max_lr, mode='triangular')]
        optimizer = [self.vae.optimizer]

        lr_and_opt = zip(opt_scheduler, optimizer)
        scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
        train_data = self.train_data.batch(self.batch_size)
        history = self.vae.fit(train_data, epochs=self.epochs, callbacks=[scheduler])

        tolerance = 1e-6

        for lr in history.history['lr']:
            self.assertGreaterEqual(lr, self.base_lr)
            self.assertLessEqual(lr, self.max_lr + tolerance)

    def test_multi_opt_lr_scheduler_with_decay(self):
        self.vae.compile()
        opt_scheduler = [ExponentialDecayScheduler(self.base_lr, self.decay_rate, self.step_size)]
        optimizer = [self.vae.optimizer]

        lr_and_opt = zip(opt_scheduler, optimizer)
        scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
        train_data = self.train_data.batch(self.batch_size)
        history = self.vae.fit(train_data, epochs=self.epochs, callbacks=[scheduler])

        tolerance = 1e-6

        for lr in history.history['lr']:
            self.assertGreaterEqual(lr, self.base_lr * (self.decay_rate ** self.epochs / self.step_size))
            self.assertLessEqual(lr, self.base_lr + tolerance)


class TestVAEUtils(unittest.TestCase):
    def setUp(self):
        cur = os.getcwd()
        filepath = os.path.join(cur, '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
        self.train_data, self.val_data, self.test_data, _ = generate_data_thickness_only(filepath)
        self.input_dim = self.train_data.element_spec[0].shape[0]
        self.batch_size = 128
        self.epochs = 10
        self.encoder = create_vae_encoder(self.input_dim, [100, 100], 20)
        self.decoder = create_vae_decoder(20, [100, 100], self.input_dim)
        self.vae = VAE(self.encoder, self.decoder)
        self.base_lr = 0.001
        self.max_lr = 0.006
        self.step_size = 20
        self.decay_rate = 0.9

    def test_save_vae(self):
        self.vae.compile()
        savefile = os.path.join(os.getcwd(), '../outputs/models/vae/test')
        save_vae(self.vae, savefile)
        self.assertTrue(os.path.exists(savefile + '/encoder'))
        self.assertTrue(os.path.exists(savefile + '/decoder'))

        if os.path.exists(savefile):
            shutil.rmtree(savefile)

    def test_load_vae(self):
        savefile = os.path.join(os.getcwd(), '../outputs/models/vae/test')
        self.vae.compile()
        save_vae(self.vae, savefile)
        loaded_vae = load_vae(savefile)
        og_config = self.vae.get_config()
        loaded_config = loaded_vae.get_config()

        self.assertEqual(og_config['encoder']['layers'], loaded_config['encoder']['layers'])
        self.assertEqual(og_config['decoder']['layers'], loaded_config['decoder']['layers'])

        if os.path.exists(savefile):
            shutil.rmtree(savefile)

    def test_create_vae(self):
        h_dim = [100, 100]
        z_dim = 20
        vae = create_vae(self.input_dim, h_dim, z_dim)
        train_batches = self.train_data.batch(self.batch_size)
        data = next(iter(train_batches))
        z_mean, z_log_var, z, x_hat = self.vae(data)
        z_mean_c, z_log_var_c, z_c, x_hat_c = vae(data)
        self.assertEqual(z_mean.shape, z_mean_c.shape)
        self.assertEqual(z_log_var.shape, z_log_var_c.shape)
        self.assertEqual(z.shape, z_c.shape)
        self.assertEqual(x_hat.shape, x_hat_c.shape)

    def test_train_val_vae(self):
        self.vae.compile()
        train_batched = self.train_data.batch(self.batch_size)
        val_batch_size = self.val_data.cardinality().numpy()
        val_batched = self.val_data.batch(val_batch_size)
        self.vae, hist = train_val_vae(self.vae, train_batched, val_batched, epochs=self.epochs)
        self.assertIn('total_loss', hist.history)
        self.assertIn('reconstruction_loss', hist.history)
        self.assertIn('kl_loss', hist.history)
        self.assertIn('val_total_loss', hist.history)
        self.assertIn('val_reconstruction_loss', hist.history)
        self.assertIn('val_kl_loss', hist.history)
        self.assertIn('lr', hist.history)
        self.assertLessEqual(len(hist.history['total_loss']), self.epochs)
        self.assertTrue(True)

    def test_generate_metrics(self):
        self.vae.compile()
        train_batched = self.train_data.batch(self.batch_size)
        val_batch_size = self.val_data.cardinality().numpy()
        val_batched = self.val_data.batch(val_batch_size)
        self.vae.fit(train_batched, epochs=self.epochs)
        hist = self.vae.evaluate(val_batched, return_dict=True)
        self.assertIn('r2', hist)
        self.assertIn('r2_feat', hist)
        self.assertEqual(len(hist['r2']), val_batch_size)
        self.assertEqual(len(hist['r2_feat']), self.input_dim)

    def test_train_val_vae_with_lr_scheduler(self):
        self.vae.compile()
        opt_scheduler = [CyclicLR(self.base_lr, self.step_size, self.max_lr, mode='triangular')]
        optimizer = [self.vae.optimizer]

        lr_and_opt = zip(opt_scheduler, optimizer)
        scheduler = MultiOptimizerLearningRateScheduler(lr_and_opt)
        self.vae, hist = train_val_vae(self.vae, self.train_data, self.val_data, self.batch_size, self.epochs,
                                       lr_scheduler=scheduler)
        self.assertIn('total_loss', hist.history)
        self.assertIn('reconstruction_loss', hist.history)
        self.assertIn('kl_loss', hist.history)
        self.assertIn('val_total_loss', hist.history)
        self.assertIn('val_reconstruction_loss', hist.history)
        self.assertIn('val_kl_loss', hist.history)
        self.assertLessEqual(len(hist.history['total_loss']), self.epochs)
        tolerance = 1e-6
        for lr in hist.history['lr']:
            self.assertGreaterEqual(lr, self.base_lr)
            self.assertLessEqual(lr, self.max_lr + tolerance)
        self.assertTrue(True)

    def test_train_val_vae_with_savefile(self):
        self.vae.compile()
        savefile = os.path.join(os.getcwd(), '../outputs/models/vae/test')
        train_batched = self.train_data.batch(self.batch_size)
        val_batch_size = self.val_data.cardinality().numpy()
        val_batched = self.val_data.batch(val_batch_size)
        self.vae, hist = train_val_vae(self.vae, train_batched, val_batched, epochs=self.epochs, savefile=savefile)
        self.assertIn('total_loss', hist.history)
        self.assertIn('reconstruction_loss', hist.history)
        self.assertIn('kl_loss', hist.history)
        self.assertIn('val_total_loss', hist.history)
        self.assertIn('val_reconstruction_loss', hist.history)
        self.assertIn('val_kl_loss', hist.history)
        self.assertLessEqual(len(hist.history['total_loss']), self.epochs)
        self.assertTrue(os.path.exists(savefile + '/encoder'))
        self.assertTrue(os.path.exists(savefile + '/decoder'))

        if os.path.exists(savefile):
            shutil.rmtree(savefile)

    def test_cross_validate_simple(self):
        param_grid = create_param_grid([[100, 100]], [20], [0.2], ['relu'], ['glorot_uniform'], betas=[.001])
        self.vae.compile()
        save_path = os.path.join(os.getcwd(), '../outputs/models/vae/')
        cv = VAECrossValidator(param_grid, self.input_dim, k_folds=5, save_path=save_path)
        results = cv.cross_validate(self.train_data, epochs=10)
        (params, metrics) = results[0]

        filename = get_filename_from_params(param_grid[0], epochs=10)
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename)))
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename + '/encoder')))
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename + '/decoder')))

        self.assertTrue(params['encoder']['hidden_dim'] == [100, 100])
        self.assertTrue(params['encoder']['latent_dim'] == 20)
        self.assertTrue(params['encoder']['dropout_rate'] == 0.2)
        self.assertTrue(params['encoder']['activation'] == 'relu')
        self.assertTrue(params['encoder']['initializer'] == 'glorot_uniform')

        self.assertTrue(params['decoder']['hidden_dim'] == [100, 100])
        self.assertTrue(params['decoder']['latent_dim'] == 20)
        self.assertTrue(params['decoder']['dropout_rate'] == 0.2)
        self.assertTrue(params['decoder']['activation'] == 'relu')
        self.assertTrue(params['decoder']['initializer'] == 'glorot_uniform')

        self.assertIn('Total Loss', metrics)
        self.assertIn('Reconstruction Loss', metrics)
        self.assertIn('KL Loss', metrics)
        self.assertIn('avg_val_total_losses', metrics)
        self.assertIn('avg_val_recon_losses', metrics)
        self.assertIn('avg_val_kl_losses', metrics)
        self.assertIn('avg_training_losses', metrics)
        self.assertIn('R2', metrics)

        self.assertEqual(len(metrics['avg_val_total_losses']), 10)

        if os.path.exists(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename)):
            shutil.rmtree(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename))

    def test_cross_validate_beta(self):
        param_grid = create_param_grid([[100, 100]], [20], [0.2], ['relu'], ['glorot_uniform'], [0.1, 0.5, 1.0])
        self.vae.compile()
        cv = VAECrossValidator(param_grid, self.input_dim, k_folds=5)
        results = cv.cross_validate(self.train_data, epochs=20, verbose=0)
        params = results[0][0]
        metrics = results[0][1]

        filename = get_filename_from_params(param_grid[0])
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename)))
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename + '/encoder')))
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename + '/decoder')))

        self.assertTrue(params['encoder']['hidden_dim'] == [100, 100])
        self.assertTrue(params['encoder']['latent_dim'] == 20)
        self.assertTrue(params['encoder']['dropout_rate'] == 0.2)
        self.assertTrue(params['encoder']['activation'] == 'relu')
        self.assertTrue(params['encoder']['initializer'] == 'glorot_uniform')

        self.assertTrue(params['decoder']['hidden_dim'] == [100, 100])
        self.assertTrue(params['decoder']['latent_dim'] == 20)
        self.assertTrue(params['decoder']['dropout_rate'] == 0.2)
        self.assertTrue(params['decoder']['activation'] == 'relu')
        self.assertTrue(params['decoder']['initializer'] == 'glorot_uniform')

        self.assertTrue(params['vae']['beta'] == 0.1)

        params = results[1][0]
        self.assertTrue(params['vae']['beta'] == 0.5)

        params = results[2][0]
        self.assertTrue(params['vae']['beta'] == 1.0)

        self.assertIn('total_loss', metrics)
        self.assertIn('recon_loss', metrics)
        self.assertIn('kl_loss', metrics)
        self.assertIn('avg_training_losses', metrics)

        self.assertEqual(len(metrics['avg_training_losses']), 20)

        if os.path.exists(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename)):
            shutil.rmtree(os.path.join(os.getcwd(), '../outputs/models/vae/' + filename))

    def test_cross_validate_df(self):
        # TODO: Finish this test
        param_grid = create_param_grid([[100, 100]], [20], [0.2], ['relu'], ['glorot_uniform'], betas=[.001])
        self.vae.compile()
        save_path = os.path.join(os.getcwd(), '../outputs/models/vae/')
        cv = VAECrossValidator(param_grid, self.input_dim, k_folds=5, save_path=save_path)
        results = cv.cross_validate_df(self.train_data, epochs=10)

        self.assertEqual(len(results), 5)

    def test_cross_validate_df_val(self):
        # TODO: Finish this test
        param_grid = create_param_grid([[100, 100]], [20], [0.2], ['relu'], ['glorot_uniform'], betas=[.01])
        self.vae.compile()
        save_path = os.path.join(os.getcwd(), '../outputs/models/vae/')
        cv = VAECrossValidator(param_grid, self.input_dim, k_folds=5, save_path=save_path)
        results = cv.cross_validate_df_val(self.train_data, epochs=10, val_data=self.val_data)

        self.assertEqual(len(results), 5)


class TestVAEAnalyzer(unittest.TestCase):
    def setUp(self):
        cur = os.getcwd()
        filepath = os.path.join(cur, '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
        train_data, val_data, test_data, self.feat_labels = generate_data_thickness_only(filepath)
        h_dim = [100, 100]
        self.z_dim = 20
        input_dim = train_data.element_spec[0].shape[0]
        batch_size = 128
        epochs = 10
        lr = 0.001
        encoder = create_vae_encoder(input_dim, h_dim, self.z_dim)
        decoder = create_vae_decoder(self.z_dim, h_dim, input_dim)
        vae = VAE(encoder, decoder)
        vae.compile()
        val_batch_size = val_data.cardinality().numpy()
        val_data_batched = val_data.batch(val_batch_size)
        self.train_data = train_data.batch(batch_size)
        self.hist = vae.fit(self.train_data, epochs=epochs, validation_data=val_data_batched, verbose=1)
        self.model = vae
        self.data = next(iter(val_data_batched))

    def test_create_analyzer(self):
        analyzer = VAEModelAnalyzer(self.model, self.data, self.z_dim, self.feat_labels)
        self.assertTrue(analyzer is not None)
        self.assertEqual(analyzer.z, self.z_dim)
        self.assertEqual(analyzer.val_data, self.data)
        self.assertEqual(analyzer.model, self.model)

    def test_full_stack(self):
        analyzer = VAEModelAnalyzer(self.model, self.data, self.z_dim, self.feat_labels)
        save_path = os.path.join(os.getcwd(), '../outputs/models/vae/test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        analyzer.full_stack(save_path)
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_space.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_dimensions.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'top_5_clusters.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_interpolation.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_influence.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'errors_hist.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'feature_errors.csv')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'full_stack.png')))
        shutil.rmtree(save_path)

    def test_full_stack_hist(self):
        analyzer = VAEModelAnalyzer(self.model, self.data, self.z_dim, self.feat_labels, hist=self.hist)
        save_path = os.path.join(os.getcwd(), '../outputs/models/vae/test')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        analyzer.full_stack(save_path)
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_space.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_dimensions.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'top_5_clusters.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_interpolation.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_influence.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'errors_hist.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'feature_errors.csv')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'full_stack.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'recon_loss_hist.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'kl_loss_hist.png')))

    def test_full_stack_cv(self):
        save_path = os.path.join(os.getcwd(), '../outputs/models/vae/test')
        results = generate_simple_cv(save_path)
        res = results.loc[0]
        model = load_or_train_model(save_path, res['Parameters'], self.train_data, 10)
        analyzer = VAEModelAnalyzer(model, self.data, 15, self.feat_labels, cv_results=res)
        analyzer.full_stack(save_path)
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_space.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_dimensions.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'top_5_clusters.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_interpolation.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'latent_influence.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'errors_hist.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'feature_errors.csv')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'full_stack.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'recon_loss_cv.png')))
        self.assertTrue(os.path.exists(os.path.join(save_path, 'kl_loss_cv.png')))
        shutil.rmtree(save_path)


class TestVAEVisUtils(unittest.TestCase):
    def setUp(self):
        cur = os.getcwd()
        filepath = os.path.join(cur, '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
        train_data, val_data, test_data, self.feat_labels = generate_data_thickness_only(filepath, normalize=True)
        input_dim = train_data.element_spec[0].shape[0]
        h_dim = [100, 100]
        val_batch_size = val_data.cardinality().numpy()
        val_data_batched = val_data.batch(val_batch_size)
        self.data = next(iter(val_data_batched))
        self.z_dim = 15
        encoder = create_vae_encoder(input_dim, h_dim, self.z_dim)
        decoder = create_vae_decoder(self.z_dim, h_dim, input_dim)
        vae = VAE(encoder, decoder)
        vae.compile()
        vae.fit(train_data.batch(128), epochs=10, verbose=0)
        self.model = vae

    def test_plot_latent_space(self):
        save_path = os.path.join(os.getcwd(), '../outputs/analysis/test/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'latent_space.png')
        fig = vu.visualize_latent_space(self.model, self.data, savefile=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, plt.Figure))
        ax = fig.axes[0]
        self.assertTrue(ax.get_xlabel() == "t-SNE 1")
        self.assertTrue(ax.get_ylabel() == "t-SNE 2")
        shutil.rmtree(save_path)

    def test_plot_latent_dimensions(self):
        save_path = os.path.join(os.getcwd(), '../outputs/analysis/test/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'latent_dimensions.png')
        fig = vu.plot_latent_dimensions(self.model, self.data, self.z_dim, savefile=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, plt.Figure))
        self.assertTrue(len(fig.axes) == self.z_dim)

        shutil.rmtree(save_path)

    def test_latent_clustering(self):
        save_path = os.path.join(os.getcwd(), '../outputs/analysis/test/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'clustered.png')
        fig, labels = vu.latent_clustering(self.model, self.data, 5, savefile=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, plt.Figure))
        ax = fig.axes[0]
        self.assertTrue(ax.get_xlabel() == "Component 1")
        self.assertTrue(ax.get_ylabel() == "Component 2")
        self.assertTrue(len(labels) == self.data[0].shape[0])

        shutil.rmtree(save_path)

    def test_visualize_top_clusters(self):
        save_path = os.path.join(os.getcwd(), '../outputs/analysis/test/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'top_5_clusters.png')
        fig = vu.visualize_top_clusters(self.model, self.data, 30, 5, savefile=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, plt.Figure))
        ax = fig.axes[0]
        self.assertTrue(ax.get_xlabel() == "t-SNE 1")
        self.assertTrue(ax.get_ylabel() == "t-SNE 2")
        labels = ax.get_legend().get_texts()
        self.assertTrue(len(labels) == 6)
        self.assertTrue(labels[0].get_text() == 'un-clustered')
        self.assertTrue(labels[1].get_text() == 'cluster 1')
        self.assertTrue(labels[2].get_text() == 'cluster 2')
        self.assertTrue(labels[3].get_text() == 'cluster 3')
        self.assertTrue(labels[4].get_text() == 'cluster 4')
        self.assertTrue(labels[5].get_text() == 'cluster 5')

        shutil.rmtree(save_path)

    def test_visualize_latent_influence(self):
        save_path = os.path.join(os.getcwd(), '../outputs/analysis/test/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'latent_influence.png')
        fig = vu.visualize_latent_influence(self.model, self.data, self.z_dim, savefile=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, plt.Figure))
        ax = fig.axes[0]
        self.assertTrue(ax.get_xlabel() == "Latent Dimension")
        self.assertTrue(ax.get_ylabel() == "Mean Error")
        self.assertTrue(len(ax.get_xticklabels()) == self.z_dim)

        shutil.rmtree(save_path)

    def test_visualize_latent_interpolation(self):
        save_path = os.path.join(os.getcwd(), '../outputs/analysis/test/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'latent_interpolation.png')
        fig, selected_features = vu.visualize_latent_interpolation(self.model, self.data, self.z_dim, self.feat_labels,
                                                                   num_features=6, savefile=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, plt.Figure))
        for i in range(6):
            ax = fig.axes[i]
            self.assertTrue(ax.get_xlabel() == "Latent Dimension")
            self.assertTrue(ax.get_ylabel() == "Mean Error")
            self.assertTrue(ax.get_title() == f"{self.feat_labels[selected_features[i]]}")
            self.assertTrue(len(ax.get_xticklabels()) == self.z_dim)
        shutil.rmtree(save_path)

    def test_visualize_errors_hist(self):
        save_path = os.path.join(os.getcwd(), '../outputs/analysis/test/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'errors_hist.png')
        fig = vu.visualize_errors_hist(self.model, self.data, savefile=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, plt.Figure))
        ax = fig.axes[0]
        self.assertTrue(ax.get_xlabel() == "Mean Error")
        self.assertTrue(ax.get_ylabel() == "Density")
        self.assertTrue(len(ax.get_legend().get_texts()) == 1)
        shutil.rmtree(save_path)

    def test_visualize_reconstruction_errors(self):
        save_path = os.path.join(os.getcwd(), '../outputs/analysis/test/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_file = os.path.join(save_path, 'reconstruction_errors.png')
        fig = vu.visualize_reconstruction_errors(self.model, self.data, savefile=save_file)
        self.assertTrue(os.path.exists(save_file))
        self.assertTrue(fig is not None)
        self.assertTrue(isinstance(fig, plt.Figure))
        ax = fig.axes[0]
        self.assertTrue(ax.get_xlabel() == "Original")
        self.assertTrue(ax.get_ylabel() == "Reconstruction")
        shutil.rmtree(save_path)


def generate_simple_cv(save_path):
    cur = os.getcwd()
    filepath = os.path.join(cur, '../data/cleaned_data/megasample_ctvol_500sym_max2percIV_cleaned.csv')
    train_data, val_data, test_data, feat_labels = generate_data_thickness_only(filepath)
    input_dim = train_data.element_spec[0].shape[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    param_grid = create_param_grid([[100, 100]], [15], [0.2], ['relu'], ['glorot_uniform'], betas=[.001])
    cv = VAECrossValidator(param_grid, input_dim, k_folds=5, save_path=save_path)
    results = cv.cross_validate_df_val(train_data, epochs=10, val_data=val_data)
    return results


if __name__ == '__main__':
    unittest.main()

# %%
