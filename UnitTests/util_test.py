import unittest
from utils import generate_data
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



class TestGenerateData(unittest.TestCase):
    def setUp(self) -> None:
        self.filepath = os.getcwd() + '/data/cleaned_data/megasample_cleaned.csv'
        self.raw_data = pd.read_csv(self.filepath)
        cond = self.raw_data[self.raw_data['dcode'] == 0].index
        self.train_raw = self.raw_data.loc[cond]
        self.test_raw = self.raw_data.drop(cond, axis=0, inplace=False)
    
    def test_simple_generate_data(self):
        train, val, test, labels = generate_data(self.filepath)
        train_card = train.cardinality().numpy()
        val_card = val.cardinality().numpy()
        test_card = test.cardinality().numpy()

        train_feat = train.element_spec[0].shape[0]
        val_feat = val.element_spec[0].shape[0]
        test_feat = test.element_spec[0].shape[0]

        condition_indices = self.raw_data[self.raw_data['dcode'] == 0].index
        raw_train = self.raw_data.loc[condition_indices]
        raw_test = self.raw_data.drop(condition_indices, axis=0, inplace=False)
        raw = self.raw_data.drop(['age', 'sex', 'scanner', 'euler', 'BrainSegVolNotVent', 'euler_med', 'sample', 'dcode',
                  'timepoint','lh_MeanThickness_thickness', 'rh_MeanThickness_thickness'], axis=1, inplace=False)
        self.assertTrue(train_feat == raw.shape[1])
        self.assertTrue(val_feat == raw.shape[1])
        self.assertTrue(test_feat == raw.shape[1])
        self.assertTrue(len(labels) == raw.shape[1])


        self.assertTrue(train_card + val_card + test_card == raw.shape[0])
        self.assertTrue(train_card == 0.8 * raw_train.shape[0])
        self.assertTrue(val_card == 0.2 * raw_train.shape[0])
        self.assertTrue(test_card == raw_test.shape[0])

    def test_generate_data_all_features(self):
        # Test if the generated data contains all features when 'subset' is set to 'all'.
        train, val, test, labels = generate_data(self.filepath, subset='all')
        train_features = train.element_spec[0].shape[0]
        val_features = val.element_spec[0].shape[0]
        test_features = test.element_spec[0].shape[0]
        raw_features = self.raw_data.shape[1] - 11  # Excluding 11 dropped columns

        self.assertEqual(train_features, raw_features)
        self.assertEqual(val_features, raw_features)
        self.assertEqual(test_features, raw_features)

    def test_generate_data_thickness_only(self):
        # Test if the generated data contains only thickness features when 'subset' is set to 'thickness'.
        train, val, test, labels = generate_data(self.filepath, subset='thickness')
        train_features = train.element_spec[0].shape[0]
        val_features = val.element_spec[0].shape[0]
        test_features = test.element_spec[0].shape[0]
        raw = self.raw_data.drop(['lh_MeanThickness_thickness', 'rh_MeanThickness_thickness'], axis=1, inplace=False)
        raw_thickness_columns = [col for col in raw.columns if col.endswith('_thickness')]

        self.assertEqual(train_features, len(raw_thickness_columns))
        self.assertEqual(val_features, len(raw_thickness_columns))
        self.assertEqual(test_features, len(raw_thickness_columns))

    def test_generate_data_volume_only(self):
        # Test if the generated data contains only volume features when 'subset' is set to 'volume'.
        train, val, test, labels = generate_data(self.filepath, subset='volume')
        train_features = train.element_spec[0].shape[0]
        val_features = val.element_spec[0].shape[0]
        test_features = test.element_spec[0].shape[0]
        raw_volume_columns = [col for col in self.raw_data.columns if col.endswith('_volume')]
        self.assertEqual(train_features, len(raw_volume_columns))
        self.assertEqual(val_features, len(raw_volume_columns))
        self.assertEqual(test_features, len(raw_volume_columns))

    def test_generate_data_thickness_volume(self):
        # Test if the generated data contains both thickness and volume features when 'subset' is set to 'thickness_volume'.
        train, val, test, labels = generate_data(self.filepath, subset='thickness_volume')
        train_features = train.element_spec[0].shape[0]
        val_features = val.element_spec[0].shape[0]
        test_features = test.element_spec[0].shape[0]
        raw = self.raw_data.drop(['lh_MeanThickness_thickness', 'rh_MeanThickness_thickness'], axis=1, inplace=False)
        raw_thickness_columns = [col for col in raw.columns if col.endswith('_thickness')]
        raw_volume_columns = [col for col in self.raw_data.columns if col.endswith('_volume')]
        self.assertEqual(train_features, len(raw_thickness_columns) + len(raw_volume_columns))
        self.assertEqual(val_features, len(raw_thickness_columns) + len(raw_volume_columns))
        self.assertEqual(test_features, len(raw_thickness_columns) + len(raw_volume_columns))

    def test_generate_data_invalid_subset(self):
        # Test for an invalid 'subset' argument and check if it raises a ValueError.
        with self.assertRaises(ValueError):
            generate_data(self.filepath, subset='invalid_subset')

if __name__ == '__main__':
    unittest.main()