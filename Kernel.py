from abc import abstractmethod, ABCMeta
import tensorflow.keras as keras

class BaseELM():
    """
    This class defined ELM Model interface.
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
    def fit(self, x, y):
        pass

    @abstractmethod
    def fit_batch(self, x, y, batch_size=32):
        pass

    @abstractmethod
    def predict(self, x, y):
        pass

class ELMClassifier(BaseELM):
    """
        Standard ELM Classifier
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

    def __init__(self, solver='SGD', layers=1, units=128, activation='relu', weight_initializer='normal'):
        self.__solver = self.solver_dictionary[solver]
        self.__layers = layers
        self.__hidden_units = units
        self.__model = None
        self.__activation = activation
        self.__weight_initializer = weight_initializer

        if self.__weight_initializer not in self.weight_initializer_dictionary:
            raise Exception('unexcepted weight initializer name: %s' % self.__weight_initializer)
        else:
            self.__weight_initializer = self.weight_initializer_dictionary[self.__weight_initializer]

    def fit(self, x, y, normalization=1, batch_size=32, **solver_params):

        if set(y) != {0, 1} or set(y) != {-1, 1}:
            raise Exception('ELM Binary Classifier can only fit binary classified data.\n '

                            'i.e. The label y should only contains 0, 1 or -1, 1')

        model = self.__build_model()
        model.compile(optimizer=self.__solver(solver_params),
                      loss=[keras.losses.binary_crossentropy, keras.losses.serialize],
                      loss_weights=[1, normalization],
                      metrics=[keras.metrics.binary_crossentropy])
        model.fit(x, y, batch_size=batch_size)

    def predict(self, x, y=None):
        return self.__model.predict(x)


