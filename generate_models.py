import tensorflow as tf
from models import AAE, create_encoder, create_decoder, create_discriminator, \
    discriminator_loss, autoencoder_loss, generator_loss
import math


def train_aae(data, batch_size=256, epochs=200, lr=0.0001, h_dim=None, z_dim=20):
    if h_dim is None:
        h_dim = [100, 100]
    n_features = data.element_spec[0].shape[1]
    n_labels = data.element_spec[1].shape[1]
    aae = create_aae(n_features, n_labels, h_dim, z_dim)
    aae = compile_aae(aae, lr)
    callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_decay)]
    aae.fit(data, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    return aae


def create_aae(n_features,n_labels, h_dim, z_dim):
    encoder = create_encoder(n_features, h_dim, z_dim)
    decoder = create_decoder(z_dim + n_labels, n_features, h_dim)
    discriminator = create_discriminator(z_dim, h_dim)
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
