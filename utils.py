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
    label = db.loc[['dcode']]
    enc_sex = OneHotEncoder(sparse=False)
    enc_age = OneHotEncoder(sparse=False)
    scaler = RobustScaler()
    one_hot_sex = enc_sex.fit_transform(db['sex'].values.reshape(-1, 1))
    age = db[['age']].round(0)
    one_hot_age = enc_age.fit_transform(age['age'].values.reshape(-1, 1))
    x_data = db.drop(['age', 'sex', 'scanner', 'euler', 'BrainSegVolNotVent', 'euler_med', 'sample', 'dcode',
                      'timepoint'], axis=1, inplace=False)
    y_data = np.concatenate((one_hot_age, one_hot_sex), axis=1).astype('float32')
    x_data_norm = scaler.fit_transform(x_data)
    train_x = x_data[label['dcode'] == 0]
    test_x = x_data[label['dcode'] == 1]

    return train_x, test_x

#%%
