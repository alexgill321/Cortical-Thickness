from tensorflow import keras


def encoder(n_features, h_dim, z_dim):
    """Creates the encoder."""
    inputs = keras.Input(shape=(n_features,))
    x = inputs
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    encoded = keras.layers.Dense(z_dim)(x)
    model = keras.Model(inputs=inputs, outputs=encoded)
    return model


def decoder(encoded_dim, n_features, h_dim):
    """Creates the decoder."""
    encoded = keras.Input(shape=(encoded_dim,))
    x = encoded
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    reconstruction = keras.layers.Dense(n_features, activation='linear')(x)
    model = keras.Model(inputs=encoded, outputs=reconstruction)
    return model


def discriminator(z_dim, h_dim):
    """Creates the discriminator."""
    z_features = keras.Input(shape=(z_dim,))
    x = z_features
    for n_neurons_layer in h_dim:
        x = keras.layers.Dense(n_neurons_layer)(x)
        x = keras.layers.LeakyReLU()(x)

    prediction = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=z_features, outputs=prediction)
    return model


#%%
