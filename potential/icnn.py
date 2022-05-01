"""module define a partially input convex neural network as potential function"""

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from potential.dense import Dense, Convex, FConvex


class ICNNPotential(Layer):
    """wire together the convex layers to form a partially input convex neural net"""

    def __init__(self, u_units, z_units):
        super(ICNNPotential, self).__init__(name='input_convex_nn')
        self.fu0 = Dense(u_units[0])
        self.fu1 = Dense(u_units[1])
        self.fu2 = Dense(u_units[2])

        self.fz0 = Convex(z_units[0])
        self.fz1 = Convex(z_units[1])
        self.fz2 = Convex(z_units[2])
        self.fz3 = Convex(z_units[3], activation=None)

        self.fe0 = FConvex(u_units[0])
        self.fe1 = FConvex(u_units[1])
        self.fe2 = FConvex(u_units[2])
        self.fe3 = FConvex(1, activation=None)

        self.built = True

    def xpath(self, x):
        self.x = x
        self.u1 = self.fu0(x)
        self.u2 = self.fu1(self.u1)
        self.u3 = self.fu2(self.u2)

    def call(self, inputs, back_prop=False):
        y, ye = inputs
        y = tf.expand_dims(y, -1)
        if back_prop:
            self.xpath(self.x)
        z = self.fz0([self.x, y])
        z = self.fz1([self.u1, y, z])
        z = self.fz2([self.u2, y, z])
        z = self.fz3([self.u3, y, z])

        ze = self.fe0([ye])
        ze = self.fe1([ye, ze])
        ze = self.fe2([ye, ze])
        ze = self.fe3([ye, ze])

        # return -tf.squeeze(z + y ** 2, -1), -tf.squeeze(ze, -1) - (ye[..., 0] - ye[..., 1]) ** 2
        return -tf.squeeze(z, -1), -tf.squeeze(ze, -1)
