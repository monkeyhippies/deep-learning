
    
def create_model_fn():
    sequence_length = 50
    transformer = Transformer(
        scope=0,
        num_tokens=35000,
        embedding_size=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        warmup_steps=4000,
        dropout_rate=.1,
        label_smoothing=.1
    )
    input = tf.placeholder(
        shape=[None, sequence_length],
        dtype=tf.int32
    )
    target = tf.placeholder(
        shape=[None, sequence_length],
        dtype=tf.int32
    )
    
    output, predict, loss, train = transformer(
        input,
        target
    )

    def model_fn(sess, encoder_input, decoder_input):

        return sess.run(
            output,
            feed_dict={
                input: encoder_input,
                target: decoder_input,
                transformer.is_training_holder: False
            }
        )
    
    return model_fn

b = BeamSearch(
    model_fn=create_model_fn(),
    num_hypotheses=4,
    beam_size=4,
    max_sequence_length=50,
    alpha=.6,
    eos_id=1,
    pad_id=0
)

x = np.reshape(
    np.array([[10, 100, 20, 30] + [0] * 46]),
    [1, 50]
)

test = b.search(x, True)
