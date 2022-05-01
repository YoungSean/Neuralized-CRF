# Neuralized Markov Random Field and its training and inference with tensorflow
# Author: Hao Xiong
# Email: haoxiong@outlook.com
# ============================================================================
"""module define neural networks layer as potential function"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from potential.dense import Dense


class NeuralNetPotential(Layer):
    """Parallel dense neural nets representing potential functions,
    including node potentials and edge potentials. e.g. if the
    MRF is M x N grid and number of classes is C, say C=10, then
    the input[0] node shape should be (C,M,N,1,L,batch_size,)
    the input[1] edge shape should be (C,M,N,2,L,batch_size,2) # x1, x2
    the output[0] node shape should be (C,M,N,1,L,batch_size,)
    the output[1] edge shape should be (C,M,N,2,L,batch_size,)
    """

    def __init__(self, node_units, edge_units):
        super(NeuralNetPotential, self).__init__(name='neural_net')
        """4 hidden layers for both node and edge potentials, may add dropout layer for l2 regularization"""
        act = 'selu'
        selu_ini = 'he_uniform'
        tanh_ini = 'glorot_uniform'
        self.node_dense0 = Dense(node_units[0], activation=act, kernel_initializer=selu_ini)
        self.node_dense1 = Dense(node_units[1], activation=act, kernel_initializer=selu_ini)
        self.node_dense2 = Dense(node_units[2], activation=act, kernel_initializer=selu_ini)
        self.node_dense3 = Dense(node_units[3], activation=act, kernel_initializer=selu_ini)
        self.node_dense4 = Dense(1, activation='tanh', kernel_initializer=tanh_ini)

        self.edge_dense0 = Dense(edge_units[0], activation=act, kernel_initializer=selu_ini)
        self.edge_dense1 = Dense(edge_units[1], activation=act, kernel_initializer=selu_ini)
        self.edge_dense2 = Dense(edge_units[2], activation=act, kernel_initializer=selu_ini)
        self.edge_dense3 = Dense(edge_units[3], activation=act, kernel_initializer=selu_ini)
        self.edge_dense4 = Dense(1, activation='tanh', kernel_initializer=tanh_ini)

    def call(self, inputs):
        fn = self.node_dense0(tf.expand_dims(inputs[0], -1))
        fn = self.node_dense1(fn)
        fn = self.node_dense2(fn)
        fn = self.node_dense3(fn)
        fn = tf.squeeze(self.node_dense4(fn), -1)

        fe = self.edge_dense0(inputs[1])
        fe = self.edge_dense1(fe)
        fe = self.edge_dense2(fe)
        fe = self.edge_dense3(fe)
        fe = tf.squeeze(self.edge_dense4(fe), -1)

        return fn, fe
