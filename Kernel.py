from abc import abstractmethod, ABCMeta
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as KerasKernel
import tensorflow.keras as keras

KERAS_VERBOES = True


class RBFLayer(Layer):
    """
        Implementation of RBF layer with keras
    """

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.__rbf_kernel_n]

    def __init__(self, units, rbf_units_trainable=False, **kwargs):
        self.__init_centers = None
        self.__init_radius = None
        self.__bias = None
        self.__rbf_trainable = rbf_units_trainable

        if isinstance(units, int):
            self.__rbf_kernel_n = units
        else:
            raise Exception('Only int can be set as num of rbf kernels.')

        Layer.__init__(self, **kwargs)

    def build(self, input_shape):
        _input_sample_n = input_shape[0]
        _input_sample_dimension = input_shape[1]

        self.__init_centers = self.add_weight(name='rbf_kernel_mean',
                                              shape=[self.__rbf_kernel_n, _input_sample_dimension],
                                              initializer='uniform',
                                              trainable=self.__rbf_trainable)
        self.__init_radius = self.add_weight(name='rbf_kernel_radius',
                                             shape=[self.__rbf_kernel_n],
                                             initializer=keras.initializers.truncated_normal(1, 1),
                                             trainable=self.__rbf_trainable)
        self.__bias = self.add_weight(name='rbf_bias',
                                      shape=[self.__rbf_kernel_n],
                                      initializer='uniform',
                                      trainable=self.__rbf_trainable)

    def call(self, inputs, **kwargs):

        pass


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
        raise NotImplementedError()

    @abstractmethod
    def fit_batch(self, X, y, batch_size=32):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X, y):
        raise NotImplementedError()


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

        if self.__normalization is not None:
            if self.__normalization is 'l1':
                _regularizer = keras.regularizers.l1(self.__lambda)
            if self.__normalization is 'l2':
                _regularizer = keras.regularizers.l2(self.__lambda)
        else:
            _regularizer = None
        self.__model.add(keras.layers.Dense(units=1, activation='sigmoid',
                                            bias_initializer=self.__weight_initializer,
                                            kernel_regularizer=_regularizer))
        return self.__model

    def __init__(self, solver='SGD', layers=1, units=128, activation='linear', weight_initializer='normal',
                 lr=1e-2, epochs=2, batchsize=128, momentum=0, normalization='l2', l=0.03):
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

        if normalization not in {'l2', 'l1'} and normalization is not None:
            raise Exception('invalid normalization method is given. only support l2, l1 or None as input.'
                            'your input is %s' % normalization)
        else:
            self.__normalization = normalization

        self.__lambda = l

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

