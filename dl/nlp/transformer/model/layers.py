import tensorflow as tf

class PointWiseFeedForward(tf.keras.layers.Layer):
    
    def __init__(self, num_outputs, activation_fn=tf.nn.relu):
        self.num_outputs = num_outputs
        self.activation_fn = activation_fn
        self._input_shape = None
        self._weights = None
        self._bias = None

        super(PointWiseFeedForward, self).__init__()
    def build(self, input_shape):
        self._input_shape = input_shape.as_list()

        initializer = tf.contrib.layers.xavier_initializer()

        self._weights = tf.Variable(
            initializer(
                shape=[self._input_shape[2], self.num_outputs]
            )
        )
        
        self._bias = tf.Variable(
            initializer(
                shape=[1, 1, self.num_outputs]
            )
        )
        super(PointWiseFeedForward, self).build(input_shape)

    def call(self, input):
        print(input)
        output = tf.reshape(
            tf.matmul(
                tf.reshape(input, [-1, self._input_shape[2]]),
                self._weights
            ),
            [-1, self._input_shape[1], self.num_outputs]
        )
        output += self._bias
        
        output = self.activation_fn(output)

        return output
