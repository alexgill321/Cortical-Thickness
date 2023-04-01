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
    healthy_db = db.loc[db['dcode'] == 0]
    enc_sex = OneHotEncoder(sparse=False)
    enc_age = OneHotEncoder(sparse=False)
    one_hot_sex = enc_sex.fit_transform(healthy_db['sex'].values.reshape(-1, 1))
    age = healthy_db[['age']].round(0)
    one_hot_age = enc_age.fit_transform(age['age'].values.reshape(-1, 1))
    x_data = healthy_db.drop(['age', 'sex', 'scanner', 'euler', 'BrainSegVolNotVent', 'euler_med', 'sample', 'dcode',
                              'timepoint'], axis=1, inplace=False)
    y_data = np.concatenate((one_hot_age, one_hot_sex), axis=1).astype('float32')
    scaler = RobustScaler()
    x_data_norm = scaler.fit_transform(x_data)
    n_samples = x_data.shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_data_norm, y_data))
    train_dataset = train_dataset.shuffle(buffer_size=n_samples).batch(256)
    return train_dataset

#%%
