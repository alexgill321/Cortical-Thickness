import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf


def generate_data(filepath: str, split: float = 0.2, normalize: int = 1, subset: str = 'all'):
    """
    Generate data to be used in model training. Splits the data from the specified csv file into training,
    test and validation sets

    Args:
        filepath (str): path to the csv file containing the data
        split (float): fraction of the data to be used for validation
        normalize (int): Normalization method. 0 for no normalization, 1 for standard normalization, 2 for global mean
         normalization, 3 for min-max normalization
        subset (str): Subset of the data to use. 'all' for all data, 'thickness' for thickness data only, 'volume' for
            volume data only, 'thickness_volume' for both thickness and volume data
    
    Returns: Tuple of tf.data.Dataset objects containing the training, validation and test data in that order, and the
        names of the columns in the data
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
                  'timepoint','lh_MeanThickness_thickness', 'rh_MeanThickness_thickness'], axis=1, inplace=False)

    if subset == 'thickness':
        thickness_columns = [col for col in db.columns if col.endswith('_thickness')]
        data = db[thickness_columns]
    elif subset == 'volume':
        volume_columns = [col for col in db.columns if col.endswith('_volume')]
        data = db[volume_columns]
    elif subset == 'thickness_volume':
        thickness_columns = [col for col in db.columns if col.endswith('_thickness')]
        volume_columns = [col for col in db.columns if col.endswith('_volume')]
        data = db[thickness_columns + volume_columns]
    elif subset == 'all':
        data = db
    else:
        raise ValueError("Invalid subset. Must be one of 'all', 'thickness', 'volume', 'thickness_volume'")
    
    # Create two new datasets based on the condition indices
    train_x = data.loc[condition_indices]
    test_x = data.loc[~db.index.isin(condition_indices)]

    y_data = np.concatenate((one_hot_age, one_hot_sex), axis=1).astype('float32')

    if normalize == 0:
        train_x_norm = train_x.to_numpy().astype('float64')
        test_x_norm = test_x.to_numpy().astype('float64')
    elif normalize == 1:
        scaler = StandardScaler()
        train_x_norm = scaler.fit_transform(train_x)
        test_x_norm = scaler.transform(test_x)
    elif normalize == 2:
        global_mean_train_x = train_x.values.mean()
        train_x_norm = train_x - global_mean_train_x
        test_x_norm = test_x - global_mean_train_x
    elif normalize == 3:
        scaler = MinMaxScaler()
        train_x_norm = scaler.fit_transform(train_x)
        test_x_norm = scaler.transform(test_x)
    else:
        raise ValueError("Invalid normalization method. Must be an integer from 0 - 3")

    train_y = y_data[condition_indices, :]
    test_y = y_data[~db.index.isin(condition_indices), :]

    train_x, val_x, train_y, val_y = train_test_split(train_x_norm, train_y, test_size=split, shuffle=False, random_state=42)
    val = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    train = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test = tf.data.Dataset.from_tensor_slices((test_x_norm, test_y))

    return train, val, test, data.columns

def generate_feature_names(filepath):
    db = pd.read_csv(filepath)
    db = db.drop(['age', 'sex', 'scanner', 'euler', 'BrainSegVolNotVent', 'euler_med', 'sample', 'dcode',
                  'timepoint'], axis=1, inplace=False)
    return db.columns

#%%
