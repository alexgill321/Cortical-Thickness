import tensorflow as tf
from models.aae_models import AAE, AAEEncoder, AAEDecoder, AAEDiscriminator, AAEOptimizer
from modelUtils.lr_utils import MultiOptimizerLearningRateScheduler
import os


def save_aae(aae, savefile):
    """ Save an AAE model.

    Args:
        aae (AAE): AAE model to be saved.
        savefile (str):
            Path to save the model to.
    """
    if not os.path.exists(savefile):
        os.makedirs(savefile)
    aae.encoder.save(os.path.join(savefile, 'encoder'), save_format="tf")
    aae.decoder.save(os.path.join(savefile, 'decoder'), save_format="tf")
    aae.discriminator.save(os.path.join(savefile, 'discriminator'), save_format="tf")


def load_aae(savefile):
    """ Load an AAE model from a savefile

    Loads and creates an AAE model from a savefile, returns the un-compiled model. If the model needs to be
    trained further it must be compiled with the desired optimizers, loss functions and callbacks.

    Args:
        savefile (str):
            Path to the model to load.

    Returns: AAE model loaded with the pretrained encoder, decoder and discriminator weights.
    """

    # restore encoder, decoder, discriminator from savefile
    encoder = tf.keras.models.load_model(os.path.join(savefile, 'encoder'))
    decoder = tf.keras.models.load_model(os.path.join(savefile, 'decoder'))
    discriminator = tf.keras.models.load_model(os.path.join(savefile, 'discriminator'))
    z_dim = encoder.layers[1].output_shape[1]

    # create aae instance
    aae = AAE(encoder, decoder, discriminator, z_dim)
    return aae


def create_aae(n_features, n_labels, h_dim, z_dim):
    """ Creates an AAE model with the given parameters.

    Creates an un-compiled AAE model with the given parameters. The model must be compiled with the desired
    optimizers, loss functions and callbacks before it can be trained.
    Args:
        n_features (int):
            Number of features in the input data.
        n_labels (int):
            Number of labels in the input data.
        h_dim (list):
            List of hidden layer dimensions.
        z_dim (int):
            Dimension of latent space.

    Returns: Un-compiled AAE model with the given parameters.
    """
    encoder = AAEEncoder(n_features, h_dim, z_dim)
    decoder = AAEDecoder(z_dim + n_labels, n_features, h_dim)
    discriminator = AAEDiscriminator(z_dim, h_dim)
    aae = AAE(encoder, decoder, discriminator, z_dim)
    return aae


def create_aae_scheduler(aae_optimizer, encoder_lr_scheduler=None, generator_lr_scheduler=None,
                         discriminator_lr_scheduler=None):
    """ Creates a learning rate scheduler for an AAE Optimizer.

    Creates a learning rate scheduler for an AAE Optimizer. The scheduler will update the learning rate of the
    optimizers in the AAE Optimizer object as specified by the input learning rate schedulers. If no learning rate
    scheduler is specified for an optimizer, the learning rate of that optimizer will not be assigned a scheduler.

    Args:
        aae_optimizer (AAEOptimizer):
            AAE Optimizer object to create the learning rate scheduler for.
        encoder_lr_scheduler:
            Learning rate scheduler for the encoder optimizer.
        generator_lr_scheduler:
            Learning rate scheduler for the generator optimizer.
        discriminator_lr_scheduler:
            Learning rate scheduler for the discriminator optimizer.

    Returns: Learning rate scheduler for the AAE Optimizer.
    """
    lr_schedules = []
    optimizers = []
    if encoder_lr_scheduler is not None:
        lr_schedules.append(encoder_lr_scheduler)
        optimizers.append(aae_optimizer.encoder_optimizer)
    if generator_lr_scheduler is not None:
        lr_schedules.append(generator_lr_scheduler)
        optimizers.append(aae_optimizer.generator_optimizer)
    if discriminator_lr_scheduler is not None:
        lr_schedules.append(discriminator_lr_scheduler)
        optimizers.append(aae_optimizer.discriminator_optimizer)
    lr_callback = MultiOptimizerLearningRateScheduler(lr_schedules, optimizers)
    return lr_callback


#%%
