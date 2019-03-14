!git clone https://furiousavocados19:password1234@bitbucket.org/furiousavocados19/transformer-train-input.git

import numpy as np

def batcher(filepath, batch_size, max_sequence_length=150, offset=0):

    with open(filepath, "r") as file_obj:
 
        # Skip the first "offset" rows
        for i in range(offset):
            file_obj.readline()

        for batch in file_batcher(
            file_obj,
            batch_size,
            max_sequence_length
        ):
            yield batch
\
    while True:
        with open(filepath, "r") as file_obj:
            for batch in file_batcher(
                file_obj,
                batch_size,
                max_sequence_length
            ):
                yield batch
def file_batcher(file_obj, batch_size, max_sequence_length):

    batch=[]
    for line in file_obj:
        line = np.array(
            line.strip().split(),
            dtype="uint16"
        )[: max_sequence_length]
        line = np.pad(
            line,
            (0, max_sequence_length - len(line)),
            "constant"
        )
        batch.append(line)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

            
import tensorflow as tf

class Transformer(object):
    scope_basename = "transformer_"

    def __init__(
        self,
        scope,
        num_tokens,
        embedding_size,
        num_encoder_layers,
        num_decoder_layers,
        num_heads,
        warmup_steps=4000,
        dropout_rate=.1,
        label_smoothing=.1
    ):
        
        self.output = None
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.scope = self.scope_basename + str(scope)
        self.is_training_holder = tf.placeholder(
            shape=None,
            dtype=tf.bool
        )
        self.dropout_rate = dropout_rate
        self.warmup_steps = warmup_steps
        self.label_smoothing = label_smoothing
        with tf.variable_scope(self.scope):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.embedding = Embedding(
                scope=0,
                num_tokens=self.num_tokens,
                hidden_size=self.embedding_size,
                dropout_rate=self.dropout_rate,
                is_training_holder=self.is_training_holder
            )
            self.embedding_weights_transposed = tf.transpose(
                self.embedding.embedding_weights
            )

            self.encoder = Encoder(
                num_heads=self.num_heads,
                num_layers=self.num_encoder_layers,
                scope=0,
                dropout_rate=self.dropout_rate,
                is_training_holder=self.is_training_holder
            )

            self.decoder = Decoder(
                num_heads=self.num_heads,
                num_layers=self.num_decoder_layers,
                scope=0,
                dropout_rate=self.dropout_rate,
                is_training_holder=self.is_training_holder
            )

    @property
    def _learning_rate(self):
        
        global_step = tf.cast(self.global_step, tf.float32)
        learning_rate = tf.minimum(tf.math.pow(global_step, tf.constant(-0.5, tf.float32)), global_step * tf.constant(self.warmup_steps ** -1.5, tf.float32))
        learning_rate *= tf.constant(self.embedding_size ** -0.5, tf.float32)    
        return learning_rate

    def __call__(self, inputs, targets):
        """
        @inputs: placeholder
        @targets: placeholder
        """

        #[batch_size, sequence_length]
        target_shape = targets.get_shape().as_list()
        
        encoder_output = self.embedding(inputs)
        encoder_output = self.encoder(encoder_output)
        
        decoder_output = self.embedding(targets)
        decoder_output = self.decoder(
            [encoder_output, decoder_output]
        )

        decoder_output = tf.reshape(
            tf.matmul(
                tf.reshape(
                    decoder_output,
                    [-1, self.embedding_size]
                ),
                self.embedding_weights_transposed
            ),
            [-1, target_shape[-1], self.num_tokens]
        )
        
        # [batch_size, sequence_length, num_tokens]
        decoder_output = tf.nn.softmax(
            decoder_output
        )

        self.output = decoder_output
        self.prediction = tf.argmax(self.output, -1)
        self.loss = tf.losses.softmax_cross_entropy(
            tf.one_hot(targets, self.num_tokens),
            self.output,
            label_smoothing=self.label_smoothing
        )
        
        # Subtract off minimum entropy. Taken from
        # https://github.com/tensorflow/models/blob/master/official/transformer/utils/metrics.py
        normalizing_constant = -(
            (1.0 - self.label_smoothing) * tf.log(1.0 - self.label_smoothing) + \
            (self.label_smoothing) * \
            tf.log(
              (self.label_smoothing) / (tf.to_float(self.num_tokens - 1)) + \
              1e-20
          )
        )
        self.loss -= normalizing_constant

        # ignore padded terms in loss
        self.loss = self.loss * tf.to_float(tf.not_equal(targets, 0))

        trainer = tf.train.AdamOptimizer(
            beta1=.9,
            beta2=.98,
            epsilon=10e-9,
            learning_rate=self._learning_rate
        )

        self.train = trainer.minimize(self.loss, global_step=self.global_step)
        return self.output, self.prediction, self.loss, self.train

class Embedding(tf.keras.layers.Layer):

    scope_basename = "embedding_"

    def __init__(self, scope, num_tokens, hidden_size, dropout_rate, is_training_holder):
        
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.is_training_holder = is_training_holder
        self.scope = self.scope_basename + str(scope)
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(self.scope):
            self.embedding_weights = tf.Variable(
                initializer([self.num_tokens, self.hidden_size])
            )
            self.scale_factor = tf.math.sqrt(
                tf.constant(self.hidden_size, dtype=tf.float32)
            )
        super(Embedding, self).__init__()

    def call(self, input):
        input_shape = input.get_shape().as_list()
        output = tf.one_hot(input, self.num_tokens, dtype=tf.float32)
        output = tf.reshape(
            tf.matmul(
                tf.reshape(output, [-1, self.num_tokens]),
                self.embedding_weights *( self.scale_factor)
            ),
            [-1, input_shape[-1], self.hidden_size]
        )
    
        output = output + positional_encoding(output)
        output = tf.layers.dropout(
            output,
            rate=self.dropout_rate,
            training=self.is_training_holder
        )
        
        return output

class Encoder(tf.keras.layers.Layer):
    scope_basename = "encoder_"
    def __init__(self, num_layers, num_heads, scope, dropout_rate, is_training_holder):
    
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.scope = self.scope_basename + str(scope)
        self.dropout_rate = dropout_rate
        self.is_training_holder = is_training_holder

        super(Encoder, self).__init__()

    def call(self, input):
        next_input = input
        with tf.variable_scope(self.scope):
            for i in range(self.num_layers):
                encode = Encode(
                    num_heads=8,
                    scope=i,
                    dropout_rate=self.dropout_rate,
                    is_training_holder=self.is_training_holder
                )
                next_input = encode(next_input)

        return next_input

class Decoder(tf.keras.layers.Layer):
    scope_basename = "decoder_"
    def __init__(self, num_layers, num_heads, scope, dropout_rate, is_training_holder):
    
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.scope = self.scope_basename + str(scope)
        self.dropout_rate = dropout_rate
        self.is_training_holder = is_training_holder

        super(Decoder, self).__init__()

    def call(self, inputs):
        encoder_output, decoder_input = inputs
        decoder_input = tf.pad(
            decoder_input, [[0, 0], [1, 0], [0, 0]]
        )[:,:-1,:]
        next_input = decoder_input
        with tf.variable_scope(self.scope):
            for i in range(self.num_layers):
                decode = Decode(
                    num_heads=8,
                    scope=i,
                    dropout_rate=self.dropout_rate,
                    is_training_holder=self.is_training_holder
                )
                next_input = decode([encoder_output, next_input])

        return next_input

class Decode(tf.keras.layers.Layer):
    
    scope_basename = "decode_"
    def __init__(self, num_heads, scope, dropout_rate, is_training_holder):
        
        self.num_heads = num_heads
        self.scope = self.scope_basename + str(scope)
        self.dropout_rate = dropout_rate
        self.is_training_holder = is_training_holder
        super(Decode, self).__init__()
        
    def build(self, input_shapes):
        # [batch_size, sequence_length, hidden_size]
        self.encoder_output_shape, self.decoder_input_shape = [shape.as_list() for shape in input_shapes]
        self.output_hidden_size = self.decoder_input_shape[-1]
        # I think k, q, and v all have to have same size
        self.kqv_hidden_size = self.output_hidden_size // self.num_heads

        with tf.variable_scope(self.scope):
            self.self_attention = MultiHeadAttention(
                scope="self0",
                num_heads=self.num_heads,
                kq_size=self.kqv_hidden_size,
                v_size=self.kqv_hidden_size,
                hidden_size=self.output_hidden_size,
                mask_future=True
            )

            self.normalization0 = LayerNorm()

            self.encdec_attention = MultiHeadAttention(
                scope="encdec0",
                num_heads=self.num_heads,
                kq_size=self.kqv_hidden_size,
                v_size=self.kqv_hidden_size,
                hidden_size=self.output_hidden_size,
                mask_future=False
            )

            self.normalization1 = LayerNorm()

            self.feed_forward0 = PointWiseFeedForward(
                scope=0,
                num_outputs=self.output_hidden_size * 2
            )
            self.feed_forward1 = PointWiseFeedForward(
                scope=1,
                num_outputs=self.output_hidden_size
            )
    
            self.normalization2 = LayerNorm()

            super(Decode, self).build(input_shapes)

    def call(self, inputs):
        encoder_output, decode_input = inputs
        residual = decode_input
        output = self.self_attention(decode_input, decode_input)
        output = tf.layers.dropout(
            output,
            rate=self.dropout_rate,
            training=self.is_training_holder
        )
        output = self.normalization0(residual + output)
        residual = output
        output = self.encdec_attention(encoder_output, decode_input)
        output = tf.layers.dropout(
            output,
            rate=self.dropout_rate,
            training=self.is_training_holder
        )
        output = self.normalization1(residual + output)
        residual = output
        output = self.feed_forward0(output)
        output = self.feed_forward1(output)
        output = tf.layers.dropout(
            output,
            rate=self.dropout_rate,
            training=self.is_training_holder
        )
        output = self.normalization2(residual + output)
        
        return output

class Encode(tf.keras.layers.Layer):
    
    scope_basename = "encode_"
    def __init__(self, num_heads, scope, dropout_rate, is_training_holder):
        
        self.num_heads = num_heads
        self.scope = self.scope_basename + str(scope)
        self.dropout_rate = dropout_rate
        self.is_training_holder = is_training_holder
        super(Encode, self).__init__()
        
    def build(self, input_shape):
        
        # [batch_size, sequence_length, hidden_size]
        self.input_size = input_shape.as_list()
        self.output_hidden_size = self.input_size[-1]
        self.kqv_hidden_size = self.output_hidden_size // self.num_heads

        with tf.variable_scope(self.scope):
            self.attention = MultiHeadAttention(
                scope=0,
                num_heads=self.num_heads,
                kq_size=self.kqv_hidden_size,
                v_size=self.kqv_hidden_size,
                hidden_size=self.output_hidden_size,
                mask_future=False
            )
            
            self.normalization0 = LayerNorm()
            self.feed_forward0 = PointWiseFeedForward(
                scope=0,
                num_outputs=self.output_hidden_size * 2
            )
            self.feed_forward1 = PointWiseFeedForward(
                scope=1,
                num_outputs=self.output_hidden_size
            )
    
            self.normalization1 = LayerNorm()

            super(Encode, self).build(input_shape)

    def call(self, input):
        residual = input
        output = self.attention(input, input)
        output = tf.layers.dropout(
            output,
            rate=self.dropout_rate,
            training=self.is_training_holder
        )
        output = self.normalization0(residual + output)
        residual = output
        output = self.feed_forward0(output)
        output = self.feed_forward1(output)
        output = tf.layers.dropout(
            output,
            rate=self.dropout_rate,
            training=self.is_training_holder
        )
        output = self.normalization1(residual + output)
        
        return output

class MultiHeadAttention(tf.keras.layers.Layer):
    
    scope_basename = "attention_"
    def __init__(self, scope, num_heads, kq_size, v_size, hidden_size, mask_future=False):
        
        self.scope = self.scope_basename + str(scope)
        self.num_heads = num_heads
        self.kq_size = kq_size
        self.kq_size_tensor = tf.constant(self.kq_size, dtype=tf.float32)
        self.v_size = v_size
        self.hidden_size = hidden_size
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.mask_future = mask_future
    
        super(MultiHeadAttention, self).__init__()

    def split_heads(self, input, sequence_length, input_dims):
        
        return tf.reshape(
            tf.transpose(
                tf.reshape(
                    input,
                    [-1, sequence_length, self.num_heads, input_dims // self.num_heads]
                ),
                [2, 0, 1, 3]
            ),
            [self.num_heads, -1, input_dims // self.num_heads]
        )

    def call(self, kv_input, q_input):

        #kv_input [batch_size, sequence_length, kv_dim]
        #q_input [batch_size, sequence_length, q_dim]
        kv_input_shape = kv_input.get_shape().as_list()
        q_input_shape = q_input.get_shape().as_list()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            k_weights = tf.get_variable(
                name="k_weights",
                shape=[self.num_heads, kv_input_shape[-1] // self.num_heads, self.kq_size],
                initializer=self.initializer
            )

            v_weights = tf.get_variable(
                name="v_weights",
                shape=[self.num_heads, kv_input_shape[-1] // self.num_heads, self.v_size],
                initializer=self.initializer
            )

            q_weights = tf.get_variable(
                name="q_weights",
                shape=[self.num_heads, q_input_shape[-1] // self.num_heads, self.kq_size],
                initializer=self.initializer
            )

            o_weights = tf.get_variable(
                name="o_weights",
                shape=[self.num_heads * self.v_size, self.hidden_size],
                initializer=self.initializer
            )

        #[batch_size, num_heads, sequence_length, kq_size]
        k = tf.matmul(self.split_heads(kv_input, kv_input_shape[-2], kv_input_shape[-1]), k_weights)
        k = tf.reshape(
            k,
            [self.num_heads, -1, kv_input_shape[-2], self.kq_size]
        )
        k = tf.transpose(
            k,
            [1, 0, 2, 3]
        )
        #[batch_size, num_heads, sequence_length, kq_size]
        v = tf.matmul(self.split_heads(kv_input, kv_input_shape[-2], kv_input_shape[-1]), v_weights)
        v = tf.reshape(
            v,
            [self.num_heads, -1, kv_input_shape[-2], self.v_size]
        )
        v = tf.transpose(
            v,
            [1, 0, 2, 3]
        )

        # [batch_size, num_heads, sequence_length, kq_size]
        q = tf.matmul(self.split_heads(q_input, q_input_shape[-2], q_input_shape[-1]), q_weights)
        q = tf.reshape(
            q,
            [self.num_heads, -1, q_input_shape[-2], self.kq_size]
        )
        q = tf.transpose(
            q,
            [1, 0, 2, 3]
        )
        
        # (QK_t)V/sqrt(k_size)
        # [batch_size, num_Heads, sequence_length, v_size]
        qk = tf.matmul(
            q,
            tf.transpose(k, [0, 1, 3, 2])
        ) // tf.math.sqrt(self.kq_size_tensor)

        if self.mask_future:
            qk = tf.matrix_band_part(qk, -1, 0)

        heads = tf.matmul(
            tf.nn.softmax(
                qk,
                axis=-1
            ),
            v
        )
        
        # [batch_size * sequence_length, num_Heads * v_size]
        heads = tf.reshape(
            tf.transpose(
                heads,
                [0, 2, 1, 3]
            ),
            [-1, self.num_heads * self.v_size]
        )
        
        # [batch_size, sequence_length, hidden_size]
        return tf.reshape(
            tf.matmul(heads, o_weights),
            [-1, kv_input_shape[-2], self.hidden_size]
        )

class LayerNorm(tf.keras.layers.Layer):
    
    def __init__(self):
        
        super(LayerNorm, self).__init__()
    
    def build(self, input_shape):
        self.hidden_size = input_shape.as_list()[-1]
        super(LayerNorm, self).build(input_shape)
        
    def call(self, input):
        
        mean = tf.math.reduce_mean(
            input,
            axis=[-1],
            keep_dims=True
        )
        
        variance = tf.reduce_sum(
            tf.math.square(input - mean),
            axis=[-1],
            keep_dims=True
        ) / self.hidden_size

        return (input - mean) // tf.math.sqrt(variance)

class PointWiseFeedForward(tf.keras.layers.Layer):
    scope_basename = "pointwisefeedforward_"
    def __init__(self, scope, num_outputs, activation_fn=tf.nn.relu):
        self.scope = self.scope_basename + str(scope)
        self.num_outputs = num_outputs
        self.activation_fn = activation_fn
        self._input_shape = None
        self._weights = None
        self._bias = None

        super(PointWiseFeedForward, self).__init__()
    def build(self, input_shape):
        self._input_shape = input_shape.as_list()

        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope(self.scope):
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
    
def positional_encoding(input):
    input_shape = input.get_shape().as_list()
    model_size = input_shape[-1]
    position_size = input_shape[-2]

    even_indexes = tf.concat(
        [tf.reshape(
            tf.range(0, model_size, 2),
            [1, -1]
        )] * position_size,
        axis=0
    )
    num_even_indexes = even_indexes.get_shape().as_list()[-1]

    positions = tf.concat(
        [
            tf.reshape(
                tf.range(0, position_size, 1),
                [-1, 1]
            )
        ] * num_even_indexes,
        axis=1
    )

    odd_indexes = tf.concat(
        [tf.reshape(
            # If model_size is odd, this adds another col
            # which is truncated later
            tf.range(1, model_size + 1, 2),
            [1, -1]
        )] * position_size,
        axis=0
    )
    num_odd_indexes = odd_indexes.get_shape().as_list()[-1]
    
    even_encodings = tf.math.sin(
        tf.math.pow(
            tf.math.truediv(positions, 10000),
            tf.math.truediv(even_indexes, model_size)
        )
    )
    
    odd_encodings = tf.math.cos(
        tf.math.pow(
            tf.math.truediv(positions, 10000),
            tf.math.truediv(odd_indexes, model_size)
        )
    )

    encodings = tf.concat(
        [tf.cast(even_encodings, tf.float32),
         tf.cast(odd_encodings, tf.float32)
        ],
        axis=0
    )

    encodings = tf.transpose(tf.reshape(
        tf.transpose(encodings),
        [-1, position_size]
    ))
    
    if model_size % 2 == 1:
        encodings = encodings[:, :-1]
    return encodings

import os

tf.reset_default_graph()
NUM_TOKENS = 37000 # Taken from wmt.vocab file created by senetencepiece
MAX_SEQUENCE_LENGTH = 150
BATCH_SIZE = 250
SAVE_INTERVAL = 1500
NUM_SENTENCES = 4562102
RESTORE_MODEL = True
MAX_MODELS = 5
checkpoint_dir = './transformer-models'
models_dir = checkpoint_dir

checkpoint_filepath = os.path.join(
    models_dir,
    'en-de-transformer-model'
)

encoder_input = tf.placeholder(
    shape=[None, MAX_SEQUENCE_LENGTH],
    dtype=tf.int32
)

target = tf.placeholder(
    shape=[None, MAX_SEQUENCE_LENGTH],
    dtype=tf.int32
)

transformer = Transformer(
    scope=0,
    num_tokens=NUM_TOKENS,
    embedding_size=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    warmup_steps=4000,
    dropout_rate=.1
)

output, prediction, loss, train = transformer(encoder_input, target)

saver = tf.train.Saver(max_to_keep=MAX_MODELS)
with tf.Session() as sess:
    if RESTORE_MODEL:
        print("Restoring model")
        !git config --global user.email "michael_liu2@yahoo.com" && \
            git config --global user.name "Michael Liu" && \
            rm -rf transformer-models/ && \
            git clone https://furiousavocados19:password1234@bitbucket.org/furiousavocados19/transformer-models.git
        saver.restore(sess, tf.train.latest_checkpoint(models_dir))
        # Delete all old models to keep github clean
        !cd transformer-models && rm -rf *en-de-transformer-model*
        current_step = sess.run(transformer.global_step)
        offset = current_step * BATCH_SIZE % NUM_SENTENCES
    else:
        print("Starting new model")
        !rm -rf transformer-models && \
            mkdir transformer-models && \
            cd transformer-models && \
            git init && \
            git remote add origin https://furiousavocados19:password1234@bitbucket.org/furiousavocados19/transformer-models.git && \
            git config --global user.email "michael_liu2@yahoo.com" && \
            git config --global user.name "Michael Liu"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        sess.run(tf.global_variables_initializer())
        offset = 0
    english = batcher(
        "transformer-train-input/train.en",
        BATCH_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        offset=offset
    )

    german = batcher(
        "transformer-train-input/train.en",
        BATCH_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        offset=offset
    )

    for e_batch, g_batch in zip(english, german):
        loss_output, _, step_output = sess.run(
            [loss, train, transformer.global_step], 
            {
                encoder_input: e_batch,
                target: g_batch,
                transformer.is_training_holder: True
            }
        )
        if step_output % 25 == 0:
            loss_output = np.mean(loss_output)
            print("Iteration: {} Loss: {}".format(step_output, loss_output))
        if step_output % SAVE_INTERVAL == 0:
            saver.save(
                sess, checkpoint_filepath,
                global_step=step_output
            )
            !cd transformer-models && \
                git add -u && \
                git add -A && \
                git commit -m 'Commit' && \
                git push -f origin master;

