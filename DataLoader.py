import pickle as pkl
import pandas as pd
from sklearn.datasets.samples_generator import make_classification, make_regression


def mount_from_file(f_path):

    if '.pkl' in f_path:
        with open(f_path, 'rb') as file:
            return pkl.load(file)

    if '.csv' in f_path:
        return pd.read_csv(f_path)

    raise Exception('file type that can not be parsed.')


def generate_random_dataset(data_type, amount=500, features=10):

    _valid_datatype = {
        'Normal Distribution Classified': make_classification(n_features=features, )
    }

    return None
