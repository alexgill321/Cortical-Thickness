import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import tensorflow as tf


def generate_data(filepath):
    """
    Generate data to be used in model training

    Separates data into features and labels,
     and splits into sets based on the value in dcode
    """
    db = pd.read_csv(filepath)
    enc_sex = OneHotEncoder(sparse=False)
    enc_age = OneHotEncoder(sparse=False)
    scaler = RobustScaler()
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
    train_x_norm = scaler.fit_transform(train_x)
    test_x_norm = scaler.fit_transform(test_x)
    train_y = y_data[condition_indices, :]
    test_y = y_data[~db.index.isin(condition_indices), :]

    train = tf.data.Dataset.from_tensor_slices((train_x_norm, train_y))
    test = tf.data.Dataset.from_tensor_slices((test_x_norm, test_y))
    return train, test


def generate_data_validation(filepath, validation_split=0.2):
    """
    Generate data to be used in model training

    splits the data from the csv file into training test and validation sets
    """
    train, test = generate_data(filepath)
    train, validation = tf.keras.utils.split_data(train, validation_split)

    return train, validation, test
#%%
