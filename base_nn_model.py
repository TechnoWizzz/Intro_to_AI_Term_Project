import tensorflow as tf

class DDDQN(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.d2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.v = tf.keras.layers.Dense(1, activation=None)
        self.a = tf.keras.layers.Dense(num_of_actions, activation=None)
        
    def call(self, input):
        x = self.d1(input)
        x = self.d2(x)
        v = self.v(x)
        a = self.a(x)
        q_val = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        return q_val