import tensorflow as tf
from models import AAE, Encoder, Decoder, Discriminator, \
    discriminator_loss, autoencoder_loss, generator_loss
import os
import math


def train_aae(data, batch_size=256, epochs=200, lr=0.0001, h_dim=None, z_dim=20, savefile=None):
    data = data.batch(batch_size)
    if h_dim is None:
        h_dim = [100, 100]
    n_features = data.element_spec[0].shape[1]
    n_labels = data.element_spec[1].shape[1]
    aae = create_aae(n_features, n_labels, h_dim, z_dim)
    aae = compile_aae(aae, lr)
    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_decay)]
    aae.fit(data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    if savefile is not None:
        if not os.path.exists(savefile):
            os.makedirs(savefile)
        aae.encoder.save(os.path.join(savefile, 'encoder'), save_format="tf")
        aae.decoder.save(os.path.join(savefile, 'decoder'), save_format="tf")
        aae.discriminator.save(os.path.join(savefile, 'discriminator'), save_format="tf")


def test_aae(data, savefile):
    # restore encoder, decoder, discriminator from savefile
    encoder = tf.keras.models.load_model(os.path.join(savefile, 'encoder'))
    decoder = tf.keras.models.load_model(os.path.join(savefile, 'decoder'))
    discriminator = tf.keras.models.load_model(os.path.join(savefile, 'discriminator'))
    z_dim = encoder.layers[1].output_shape[1]
    # create aae instance
    aae = AAE(encoder, decoder, discriminator, z_dim)
    # run test data through aae, saving resulting loss values for each sample
    output = []
    for point in data:
        output.append(aae(point))
    return output


def create_aae(n_features, n_labels, h_dim, z_dim):
    encoder = Encoder(n_features, h_dim, z_dim)
    decoder = Decoder(z_dim + n_labels, n_features, h_dim)
    discriminator = Discriminator(z_dim, h_dim)
    aae = AAE(encoder, decoder, discriminator, z_dim)
    return aae


def compile_aae(aae, base_lr):
    ae_optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
    dc_optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
    aae.compile(ae_optimizer, dc_optimizer, gen_optimizer,
                generator_loss, autoencoder_loss, discriminator_loss)
    return aae


def lr_decay(epoch, lr):
    max_lr = 0.005  # maximum learning rate
    decay_rate = 0.1  # decay rate
    return lr * math.exp(-decay_rate*epoch) if lr * math.exp(-decay_rate*epoch) > max_lr else max_lr

#%%
