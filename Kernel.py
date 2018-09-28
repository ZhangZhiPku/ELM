from abc import abstractmethod, ABCMeta
import tensorflow.keras as keras

KERAS_VERBOES = False

class BaseELM():
    """
        This class defined ELM Model interface.

        !DO NOT MODIFY THIS CLASS!
    """

    def __init__(self):
        self.solver_dictionary = {
            'SGD': keras.optimizers.SGD,
            'Adam': keras.optimizers.Adam,
            'Newton': None,
            'NormalEquation': None
        }

        self.weight_initializer_dictionary = {
            'normal': keras.initializers.normal(mean=0.0, stddev=1.0),
            'uniform': keras.initializers.uniform(minval=0.0, maxval=1.0),
            'random': keras.initializers.random_uniform(minval=0.0, maxval=1.0)
        }

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def fit_batch(self, X, y, batch_size=32):
        pass

    @abstractmethod
    def predict(self, X, y):
        pass


class ELMClassifier(BaseELM):
    """
        Standard ELM Classifier

        This implementation of ELM is based on tensorflow 1.9.0 with keras

        it will use SGD to solve the ELM problem with L2 normalization params.

        reference of ELM algorithm can be found at here:

            Extreme learning machine: theory and applications
    """
    def __build_model(self):
        self.__model = keras.models.Sequential()

        for layers_n in range(self.__layers):
            _hidden_layer = keras.layers.Dense(units=self.__hidden_units, activation=self.__activation,
                                               bias_initializer=self.__weight_initializer)
            self.__model.add(_hidden_layer)
            _hidden_layer.trainable = False

        self.__model.add(keras.layers.Dense(units=1, activation='sigmoid', bias_initializer=self.__weight_initializer))
        return self.__model

    def __init__(self, solver='SGD', layers=1, units=128, activation='linear', weight_initializer='normal',
                 lr=1e-2, epochs=2, batchsize=128, momentum=0):
        BaseELM.__init__(self)
        self.__fit_epochs = epochs
        self.__fit_batchsize = batchsize
        self.__solver = keras.optimizers.SGD(lr=lr, momentum=momentum)
        self.__layers = layers
        self.__hidden_units = units
        self.__model = None
        self.__activation = activation
        self.__weight_initializer = weight_initializer

        if self.__weight_initializer not in self.weight_initializer_dictionary:
            raise Exception('unexcepted weight initializer name: %s' % self.__weight_initializer)
        else:
            self.__weight_initializer = self.weight_initializer_dictionary[self.__weight_initializer]

    def fit(self, X, y):

        if len(set(y)) is not 2:
            raise Exception('ELM Binary Classifier can only fit binary classified data. '
                            'i.e. The label y should only contains 0, 1 or -1, 1')

        model = self.__build_model()
        model.compile(optimizer=self.__solver,
                      loss=keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.binary_crossentropy])
        model.fit(X, y, batch_size=self.__fit_batchsize, epochs=self.__fit_epochs, verbose=KERAS_VERBOES)

    def predict(self, X, y=None):
        return self.__model.predict(X)


class ELMRegressor(BaseELM):
    """
        Standard ELM Regressor for solving regression problem

        This implementation is based on tensorflow 1.9.0 with keras

        it will use SGD to solve this ELM problem with L2 normalization term.

        reference can be found here:

            Extreme learning machine: theory and applications
    """

    def fit(self, X, y):
        pass

    def fit_batch(self, X, y, batch_size=32):
        pass

    def predict(self, X, y):
        pass

