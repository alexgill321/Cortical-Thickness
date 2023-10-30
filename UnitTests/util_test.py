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
        self.raw_data.drop(['age', 'sex', 'scanner', 'euler', 'BrainSegVolNotVent', 'euler_med', 'sample', 'dcode',
                  'timepoint','lh_MeanThickness_thickness', 'rh_MeanThickness_thickness'], axis=1, inplace=True)
        self.train_raw = self.raw_data.loc[cond]
        self.test_raw = self.raw_data.drop(cond, axis=0, inplace=False)
    
    def test_simple(self):
        train, val, test, labels = generate_data(self.filepath)
        train_card = train.cardinality().numpy()
        val_card = val.cardinality().numpy()
        test_card = test.cardinality().numpy()

        train_feat = train.element_spec[0].shape[0]
        val_feat = val.element_spec[0].shape[0]
        test_feat = test.element_spec[0].shape[0]

        self.assertTrue(train_feat == self.train_raw.shape[1])
        self.assertTrue(val_feat == self.train_raw.shape[1])
        self.assertTrue(test_feat == self.train_raw.shape[1])
        self.assertTrue(len(labels) == self.train_raw.shape[1])

        self.assertTrue(train_card + val_card + test_card == self.raw_data.shape[0])
        self.assertTrue(train_card == 0.8 * self.train_raw.shape[0])
        self.assertTrue(val_card == 0.2 * self.train_raw.shape[0])
        self.assertTrue(test_card == self.test_raw.shape[0])

    def test_all_features(self):
        # Test if the generated data contains all features when 'subset' is set to 'all'.
        train, val, test, labels = generate_data(self.filepath, subset='all')
        train_features = train.element_spec[0].shape[0]
        val_features = val.element_spec[0].shape[0]
        test_features = test.element_spec[0].shape[0]
        raw_features = self.raw_data.shape[1]  # Excluding 11 dropped columns

        self.assertEqual(train_features, raw_features)
        self.assertEqual(val_features, raw_features)
        self.assertEqual(test_features, raw_features)

    def test_thickness_only(self):
        # Test if the generated data contains only thickness features when 'subset' is set to 'thickness'.
        train, val, test, labels = generate_data(self.filepath, subset='thickness')
        train_features = train.element_spec[0].shape[0]
        val_features = val.element_spec[0].shape[0]
        test_features = test.element_spec[0].shape[0]
        raw_thickness_columns = [col for col in self.train_raw.columns if col.endswith('_thickness')]

        self.assertEqual(train_features, len(raw_thickness_columns))
        self.assertEqual(val_features, len(raw_thickness_columns))
        self.assertEqual(test_features, len(raw_thickness_columns))

    def test_volume_only(self):
        # Test if the generated data contains only volume features when 'subset' is set to 'volume'.
        train, val, test, labels = generate_data(self.filepath, subset='volume')
        train_features = train.element_spec[0].shape[0]
        val_features = val.element_spec[0].shape[0]
        test_features = test.element_spec[0].shape[0]
        raw_volume_columns = [col for col in self.raw_data.columns if col.endswith('_volume')]
        self.assertEqual(train_features, len(raw_volume_columns))
        self.assertEqual(val_features, len(raw_volume_columns))
        self.assertEqual(test_features, len(raw_volume_columns))

    def test_thickness_volume(self):
        # Test if the generated data contains both thickness and volume features when 'subset' is set to 'thickness_volume'.
        train, val, test, labels = generate_data(self.filepath, subset='thickness_volume')
        train_features = train.element_spec[0].shape[0]
        val_features = val.element_spec[0].shape[0]
        test_features = test.element_spec[0].shape[0]
        raw_thickness_columns = [col for col in self.train_raw.columns if col.endswith('_thickness')]
        raw_volume_columns = [col for col in self.train_raw.columns if col.endswith('_volume')]

        self.assertEqual(train_features, len(raw_thickness_columns) + len(raw_volume_columns))
        self.assertEqual(val_features, len(raw_thickness_columns) + len(raw_volume_columns))
        self.assertEqual(test_features, len(raw_thickness_columns) + len(raw_volume_columns))

    def test_invalid_subset(self):
        # Test for an invalid 'subset' argument and check if it raises a ValueError.
        with self.assertRaises(ValueError):
            generate_data(self.filepath, subset='invalid_subset')

    def test_all_no_norm(self):
        # Test if the generated data contains all features when 'subset' is set to 'all' and 'norm' is set to 'none'.
        train, val, test, labels = generate_data(self.filepath, subset='all', normalize=0)
        train_x, val_x = train_test_split(self.train_raw.to_numpy(), test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = self.test_raw.to_numpy()[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_all_standard(self):
        train, val, test, labels = generate_data(self.filepath, subset='all', normalize=1)
        scaler = StandardScaler()
        train_norm = scaler.fit_transform(self.train_raw.to_numpy())
        test_norm = scaler.transform(self.test_raw.to_numpy())
        train_x, val_x = train_test_split(train_norm, test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = test_norm[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_all_minmax(self):
        train, val, test, labels = generate_data(self.filepath, subset='all', normalize=3)
        scaler = MinMaxScaler()
        train_norm = scaler.fit_transform(self.train_raw.to_numpy())
        test_norm = scaler.transform(self.test_raw.to_numpy())
        train_x, val_x = train_test_split(train_norm, test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = test_norm[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_volume_no_norm(self):
        train, val, test, labels = generate_data(self.filepath, subset='volume', normalize=0)
        vol_columns = [col for col in self.train_raw.columns if col.endswith('_volume')]
        train_x, val_x = train_test_split(self.train_raw[vol_columns].to_numpy().astype('float64'), test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = self.test_raw[vol_columns].to_numpy().astype('float64')[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type)
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_volume_standard(self):
        train, val, test, labels = generate_data(self.filepath, subset='volume', normalize=1)
        scaler = StandardScaler()
        vol_columns = [col for col in self.train_raw.columns if col.endswith('_volume')]
        train_norm = scaler.fit_transform(self.train_raw[vol_columns].to_numpy().astype('float64'))
        test_norm = scaler.transform(self.test_raw[vol_columns].to_numpy().astype('float64'))
        train_x, val_x = train_test_split(train_norm, test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = test_norm[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")
    
    def test_volume_minmax(self):
        train, val, test, labels = generate_data(self.filepath, subset='volume', normalize=3)
        scaler = MinMaxScaler()
        vol_columns = [col for col in self.train_raw.columns if col.endswith('_volume')]
        train_norm = scaler.fit_transform(self.train_raw[vol_columns].to_numpy().astype('float64'))
        test_norm = scaler.transform(self.test_raw[vol_columns].to_numpy().astype('float64'))
        train_x, val_x = train_test_split(train_norm, test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = test_norm[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_thickness_no_norm(self):
        train, val, test, labels = generate_data(self.filepath, subset='thickness', normalize=0)
        thickness_columns = [col for col in self.train_raw.columns if col.endswith('_thickness')]
        train_x, val_x = train_test_split(self.train_raw[thickness_columns].to_numpy().astype('float64'), test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = self.test_raw[thickness_columns].to_numpy().astype('float64')[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")
        

    def test_thickness_standard(self):
        train, val, test, labels = generate_data(self.filepath, subset='thickness', normalize=1)
        scaler = StandardScaler()
        thickness_columns = [col for col in self.train_raw.columns if col.endswith('_thickness')]
        train_norm = scaler.fit_transform(self.train_raw[thickness_columns].to_numpy().astype('float64'))
        test_norm = scaler.transform(self.test_raw[thickness_columns].to_numpy().astype('float64'))
        train_x, val_x = train_test_split(train_norm, test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = test_norm[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_thickness_minmax(self):
        train, val, test, labels = generate_data(self.filepath, subset='thickness', normalize=3)
        scaler = MinMaxScaler()
        thickness_columns = [col for col in self.train_raw.columns if col.endswith('_thickness')]
        train_norm = scaler.fit_transform(self.train_raw[thickness_columns].to_numpy().astype('float64'))
        test_norm = scaler.transform(self.test_raw[thickness_columns].to_numpy().astype('float64'))
        train_x, val_x = train_test_split(train_norm, test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = test_norm[i]
            has_target_type = np.all(np.array([isinstance(item, np.float64) for item in out_x]))
            self.assertTrue(has_target_type) 
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_thickness_volume_no_norm(self):
        train, val, test, labels = generate_data(self.filepath, subset='thickness_volume', normalize=0)
        thickness_columns = [col for col in self.train_raw.columns if col.endswith('_thickness')]
        volume_columns = [col for col in self.train_raw.columns if col.endswith('_volume')]
        train_x, val_x = train_test_split(self.train_raw[thickness_columns + volume_columns].to_numpy().astype('float64'), test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = self.test_raw[thickness_columns + volume_columns].to_numpy().astype('float64')[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_thickness_volume_standard(self):
        train, val, test, labels = generate_data(self.filepath, subset='thickness_volume', normalize=1)
        scaler = StandardScaler()
        thickness_columns = [col for col in self.train_raw.columns if col.endswith('_thickness')]
        volume_columns = [col for col in self.train_raw.columns if col.endswith('_volume')]
        train_norm = scaler.fit_transform(self.train_raw[thickness_columns + volume_columns].to_numpy().astype('float64'))
        test_norm = scaler.transform(self.test_raw[thickness_columns + volume_columns].to_numpy().astype('float64'))
        train_x, val_x = train_test_split(train_norm, test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = test_norm[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_thickness_volume_minmax(self):
        train, val, test, labels = generate_data(self.filepath, subset='thickness_volume', normalize=3)
        scaler = MinMaxScaler()
        thickness_columns = [col for col in self.train_raw.columns if col.endswith('_thickness')]
        volume_columns = [col for col in self.train_raw.columns if col.endswith('_volume')]
        train_norm = scaler.fit_transform(self.train_raw[thickness_columns + volume_columns].to_numpy().astype('float64'))
        test_norm = scaler.transform(self.test_raw[thickness_columns + volume_columns].to_numpy().astype('float64'))
        train_x, val_x = train_test_split(train_norm, test_size=0.2, random_state=42, shuffle=False)
        train_data = train.as_numpy_iterator()
        for i, row in enumerate(train_data):
            out_x = row[0]
            exp_x = train_x[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("train data not equal")
        val_data = val.as_numpy_iterator()
        for i, row in enumerate(val_data):
            out_x = row[0]
            exp_x = val_x[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("val data not equal")
        test_data = test.as_numpy_iterator()
        for i, row in enumerate(test_data):
            out_x = row[0]
            exp_x = test_norm[i]
            self.assertTrue(np.all(np.array([isinstance(item, np.float64) for item in out_x])))
            if not np.array_equal(out_x, exp_x):
                self.fail("test data not equal")

    def test_invalid_normalize(self):
        # Test for an invalid 'normalize' argument and check if it raises a ValueError.
        with self.assertRaises(ValueError):
            generate_data(self.filepath, subset='all', normalize=4)

if __name__ == '__main__':
    unittest.main()