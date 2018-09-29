from Kernel import ELMClassifier, RBFELMClassifier
from DataLoader import mount_from_file, mount_iris_dataset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    import lightgbm as gbm
    import xgboost as xgb
except ImportError as e:
    print('You should install necessary package for comparing different models.')

from Utilities.DataProcessing import binarilize_classified, standard_scaling
from Utilities.Evaluators import binary_classified_precision

if __name__ == '__main__':

    # modify here to switch different dataset
    _data = mount_from_file('Data/creditcard.csv')
    _Y = np.array(_data['Class'])
    _X = np.array(_data.drop(['Class'], axis=1))
    _X = standard_scaling(_X)

    '''
    # iris dataset
    _data = mount_iris_dataset()
    _X = _data['data']
    _Y = _data['target'] % 2 # make it a binary classification problem
    _X = standard_scaling(_X)
    '''

    # modify here to test different Classify Model
    _model = ELMClassifier(layers=3, units=256, epochs=1,
                           batchsize=128, lr=0.01, trainable=True,
                           activation='relu')
    # _model = LogisticRegression()
    # _model = gbm.LGBMClassifier(objective='binary')
    # _model = xgb.XGBClassifier(max_depth=6, n_estimators=300)

    _kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for _itr_idx, (_train_idx, _test_idx) in enumerate(_kf.split(_X)):
        _train_x, _train_y = _X[_train_idx], _Y[_train_idx]
        _test_x, _test_y = _X[_test_idx], _Y[_test_idx]

        _model.fit(X=_train_x, y=_train_y)
        _pred_y = _model.predict(X=_test_x)

        print('Model training is finished at iteration %d. '
              'Model predict precision: %.5f' % (_itr_idx + 1, binary_classified_precision(
            y_real=_test_y, y_pred=binarilize_classified(_pred_y)
        )))

else:
    raise Exception('Please do not import this file, '
                    'only can be invoked by operating system.')