# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""A modified version of dense layer from keras.layers.Dense to do multiple flat
dense calculation simultaneously """

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations as acts
from tensorflow.keras import constraints as ctr
from tensorflow.keras import initializers as init
from tensorflow.keras import regularizers as reg


# L2REG = 1


class Convex(Layer):
    """implement the partially input convex neural net defined in https://arxiv.org/pdf/1609.07152"""

    def __init__(self, units, activation='elu', kernel_regularizer='l2', **kwargs):
        super(Convex, self).__init__(**kwargs)
        self.units = int(units)
        self.activation = acts.get(activation) if activation is not None else None
        self.kernel_initializer = init.get('he_uniform')
        self.bias_initializer = init.get('zeros')
        self.kernel_constraint = ctr.get('non_neg')  # None
        self.kernel_regularizer = reg.get(kernel_regularizer)

    def build(self, input_shape):
        # input_shape = (u_shape, y_shape, z_shape)
        out_shape = [1] * (input_shape[0].rank - 2)
        u_dim = input_shape[0][-1]
        y_dim = input_shape[1][-1]
        if len(input_shape) > 2:
            z_dim = input_shape[2][-1]
            self.W_zu = self.add_weight(name='W_zu',
                                        shape=out_shape + [u_dim, z_dim],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        # regularizer=reg.l2(l=L2REG),
                                        trainable=True)
            self.b_z = self.add_weight(name='b_z',
                                       shape=out_shape + [1, z_dim],
                                       initializer=self.bias_initializer,
                                       trainable=True)
            self.W_z = self.add_weight(name='W_z',
                                       shape=out_shape + [z_dim, self.units],
                                       initializer=self.kernel_initializer,
                                       constraint=self.kernel_constraint,
                                       regularizer=self.kernel_regularizer,
                                       # regularizer=reg.l2(l=L2REG),
                                       trainable=True)
            self.W_z.assign_sub(tf.reduce_min(self.W_z) * tf.ones_like(self.W_z))  # non-neg constraint

        self.W_yu = self.add_weight(name='W_yu',
                                    shape=out_shape + [u_dim, y_dim],
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    # regularizer=reg.l2(l=L2REG),
                                    trainable=True)
        self.b_y = self.add_weight(name='b_y',
                                   shape=out_shape + [1, y_dim],
                                   initializer=self.bias_initializer,
                                   trainable=True)
        self.W_y = self.add_weight(name='W_y',
                                   shape=out_shape + [y_dim, self.units],
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   # regularizer=reg.l2(l=L2REG),
                                   trainable=True)

        self.W_u = self.add_weight(name='W_u',
                                   shape=out_shape + [u_dim, self.units],
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   # regularizer=reg.l2(l=L2REG),
                                   trainable=True)
        self.b_u = self.add_weight(name='b_u',
                                   shape=out_shape + [1, self.units],
                                   initializer=self.bias_initializer,
                                   trainable=True)

        self.built = True

    def call(self, inputs):
        u = inputs[0]
        y = inputs[1]
        z = tf.matmul(y * (tf.matmul(u, self.W_yu) + self.b_y), self.W_y) + tf.matmul(u, self.W_u) + self.b_u
        if len(inputs) > 2:
            z += tf.matmul(inputs[2] * tf.nn.relu(tf.matmul(u, self.W_zu) + self.b_z), self.W_z)
        return self.activation(z) if self.activation is not None else z


class FConvex(Layer):
    """implement the fully input convex neural net"""

    def __init__(self, units, activation='elu', kernel_regularizer='l2', **kwargs):
        super(FConvex, self).__init__(**kwargs)
        self.units = int(units)
        self.activation = acts.get(activation) if activation is not None else None
        self.kernel_initializer = init.get('he_uniform')
        self.bias_initializer = init.get('zeros')
        self.kernel_constraint = ctr.get('non_neg')  # None
        self.kernel_regularizer = reg.get(kernel_regularizer)

    def build(self, input_shape):
        # input_shape = (y_shape, z_shape)
        out_shape = [1] * (input_shape[0].rank - 2)
        y_dim = input_shape[0][-1]
        if len(input_shape) > 1:
            z_dim = input_shape[1][-1]
            self.W_z = self.add_weight(name='W_z',
                                       shape=out_shape + [z_dim, self.units],
                                       initializer=self.kernel_initializer,
                                       constraint=self.kernel_constraint,
                                       regularizer=self.kernel_regularizer,
                                       # regularizer=reg.l2(l=L2REG),
                                       trainable=True)
            self.W_z.assign_sub(tf.reduce_min(self.W_z) * tf.ones_like(self.W_z))  # non-neg constraint

        self.W_y = self.add_weight(name='W_y',
                                   shape=out_shape + [y_dim, self.units],
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   # regularizer=reg.l2(l=L2REG),
                                   trainable=True)
        self.bias = self.add_weight(name='b',
                                    shape=out_shape + [1, self.units],
                                    initializer=self.bias_initializer,
                                    trainable=True)

        self.built = True

    def call(self, inputs):
        z = tf.matmul(inputs[0], self.W_y) + self.bias
        if len(inputs) > 1:
            z += tf.matmul(inputs[1], self.W_z)
        return self.activation(z) if self.activation is not None else z


class Dense(Layer):
    """ A N-D layer consist of many independent keras Dense layers

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Input shape:
      N-D tensor with shape: `(..., batch_size, input_dim)`.
      The most common situation would be
      a 3D input with shape `(net_num, batch_size, input_dim)`.

    Output shape:
      N-D tensor with shape: `(..., batch_size, units)`.
      For instance, for a 3D input with shape `(net_num, batch_size, input_dim)`,
      the output would have shape `(net_num, batch_size, units)`.
    """

    def __init__(self,
                 units,
                 activation='elu',
                 use_bias=True,
                 kernel_initializer='he_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer='l2',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Dense, self).__init__(
            activity_regularizer=reg.get(activity_regularizer), **kwargs)
        self.units = int(units)
        self.activation = acts.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = init.get(kernel_initializer)
        self.bias_initializer = init.get(bias_initializer)
        self.kernel_regularizer = reg.get(kernel_regularizer)
        self.bias_regularizer = reg.get(bias_regularizer)
        self.kernel_constraint = ctr.get(kernel_constraint)
        self.bias_constraint = ctr.get(bias_constraint)

    def build(self, input_shape):
        out_shape = [1] * (input_shape.rank - 2)
        self.kernel = self.add_weight(
            name='kernel',
            shape=out_shape + [input_shape[-1], self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            # regularizer=reg.l2(l=L2REG),
            constraint=self.kernel_constraint,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=out_shape + [1, self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        outputs = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'units': self.units,
            'activation': acts.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': init.serialize(self.kernel_initializer),
            'bias_initializer': init.serialize(self.bias_initializer),
            'kernel_regularizer': reg.serialize(self.kernel_regularizer),
            'bias_regularizer': reg.serialize(self.bias_regularizer),
            'activity_regularizer': reg.serialize(self.activity_regularizer),
            'kernel_constraint': ctr.serialize(self.kernel_constraint),
            'bias_constraint': ctr.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return base_config.update(config)
