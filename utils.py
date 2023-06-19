import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


def generate_data(filepath):
    """
    Generate data to be used in model training

    Separates data into features and labels,
     and splits into sets based on the value in dcode
    """
    db = pd.read_csv(filepath)
    enc_sex = OneHotEncoder(sparse_output=False)
    enc_age = OneHotEncoder(sparse_output=False)
    one_hot_sex = enc_sex.fit_transform(db['sex'].values.reshape(-1, 1))
    age = db[['age']].round(0)
    one_hot_age = enc_age.fit_transform(age['age'].values.reshape(-1, 1))

    # Get the indices matching a condition on a feature
    condition_indices = db[db['dcode'] == 0].index
    db = db.drop(['age', 'sex', 'scanner', 'euler', 'BrainSegVolNotVent', 'euler_med', 'sample', 'dcode',
                  'timepoint'], axis=1, inplace=False)

    # Create two new datasets based on the condition indices
    train_x = db.loc[condition_indices]
    test_x = db.loc[~db.index.isin(condition_indices)]

    y_data = np.concatenate((one_hot_age, one_hot_sex), axis=1).astype('float32')
    scaler = StandardScaler()
    train_x_norm = scaler.fit_transform(train_x)
    test_x_norm = scaler.transform(test_x)
    train_y = y_data[condition_indices, :]
    test_y = y_data[~db.index.isin(condition_indices), :]

    return train_x_norm, train_y, test_x_norm, test_y


def data_train_test(filepath):
    """
    Generate data to be used in model training

    splits the data from the csv file into training test and validation sets
    """

    train_x, train_y, test_x, test_y = generate_data(filepath)
    train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    return train, test


def data_validation(filepath, validation_split=0.2):
    """ Generate data to be used in model training

    splits the data from the csv file into training test and validation sets

    Args:
        filepath (str): path to the csv file containing the data
        validation_split (float): fraction of the data to be used for validation

    Returns: Tuple of tf.data.Dataset objects containing the training, validation and test data in that order.
    """
    train_x, train_y, test_x, test_y = generate_data(filepath)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=validation_split)
    train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    val = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    return train, val, test


def generate_feature_names(filepath):
    db = pd.read_csv(filepath)
    db = db.drop(['age', 'sex', 'scanner', 'euler', 'BrainSegVolNotVent', 'euler_med', 'sample', 'dcode',
                  'timepoint'], axis=1, inplace=False)
    return db.columns

#%%
