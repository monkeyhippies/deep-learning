#Implementation of "Dyanmic Routing of Capsules" https://arxiv.org/pdf/1710.09829.pdf

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

LOAD = True
if LOAD:
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(
        x_train,
        list(x_train.shape) + [1]
    )

    x_test = np.reshape(
        x_test,
        list(x_test.shape) + [1]
    )

routing_iterations = 3
capsule_dims = 8
num_digits = 10
digicap_dims = 16
y = .5

tf.reset_default_graph()
initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

def squash(tensor, axis):

    norm_2 = tf.math.reduce_sum(
        tf.math.square(tensor),
        axis=axis,
        keep_dims=True
    )

    norm = tf.math.sqrt(
        norm_2
    )

    return norm_2/(1 + norm_2) / norm * tensor

image = tf.placeholder(
  shape=[None, 28, 28, 1],
  dtype=tf.float32
)

num_images = tf.shape(image)[0]

conv1 = slim.conv2d(
  image,
  num_outputs=256,
  kernel_size=[9, 9],
  stride=[1, 1],
  activation_fn=tf.nn.relu,
  padding='VALID'
)

capsule_elements = []
for i in range(capsule_dims):
  capsule_element = slim.conv2d(
    conv1,
    num_outputs=32,
    kernel_size=[9, 9],
    stride=[2, 2],
    padding='VALID'
  )
  capsule_elements.append(capsule_element)
  
capsule_inputs = tf.concat(capsule_elements, axis=3)
capsule_outputs = squash(capsule_inputs, axis=3)
capsule_outputs = tf.reshape(capsule_outputs, [-1, 6 * 6 * 32, capsule_dims])
W = tf.Variable(
    initializer(
      shape=[6 * 6 * 32, capsule_dims, num_digits * digicap_dims],
    )
)

"""
bs = [
    tf.Variable(
        initializer(
            shape=[6 * 6 * 32,  1]
        )
    )
    for i in range(num_digits)
]

cs = [
    tf.nn.softmax(b, axis=0) for b in bs
]

"""



u = tf.transpose(
    tf.matmul(
        tf.transpose(
            capsule_outputs,
            [1, 0, 2]
        ),
        W
    ),
    [1, 0, 2]
)

u = tf.reshape(
    u,
    [-1, 6 * 6 * 32, digicap_dims, num_digits]
)

#(batch_size, num_digits, digicap_dims, num_primary_capsules)
u = tf.transpose(
    u,
    [0, 3, 2, 1]
)

"""
digicaps_inputs = tf.split(
  digicaps_inputs,
  num_digits,
  3
)
"""

b = tf.zeros(
    shape=[num_images, num_digits, 6 * 6 * 32, 1],
    dtype=tf.float32
)

for i in range(routing_iterations):
    c = tf.nn.softmax(b, 1)

    #(batch_size, num_digits, digicap_dims, 1)
    s = tf.matmul(
        u,
        c
    )

    #(batch_size, num_digits, digicap_dims, 1)
    s = tf.reshape(
        s,
        shape=[-1, num_digits, digicap_dims, 1]
    )

    v = squash(s, 2)
    b = b + tf.matmul(
        tf.transpose(
            u,
            [0, 1, 3, 2]
        ),
        v
    )
    
digicap_output = tf.reshape(
    v,
    [-1, num_digits, digicap_dims]
)

"""
digicap_outputs = []
for c, input in zip(cs, digicaps_inputs):

    input_transposed = tf.transpose(input, [0, 2, 1, 3])
    output = tf.reshape(
        tf.matmul(
            tf.reshape(
                input_transposed,
                [-1, 6 * 6 * 32]
            ),
            c
        ),
        [-1, 1, digicap_dims]
    )

    digicap_outputs.append(output)
  
  
# size (batch_size, num_digits, 16)
digicap_outputs = tf.concat(digicap_outputs, 1)
"""


target = tf.placeholder(
    shape=[None],
    dtype=tf.uint8
)
target_onehots = tf.one_hot(
    target,
    num_digits,
    dtype=tf.float32
)

digicap_norm = tf.math.sqrt(tf.math.reduce_sum(
    tf.math.square(digicap_output),
    axis=2
))

prediction = tf.argmax(digicap_norm, 1)

digicap_mask = tf.placeholder(
    dtype=tf.float32,
    shape=[None, num_digits, digicap_dims],
)

fc1 = slim.fully_connected(
    tf.contrib.layers.flatten(digicap_output * digicap_mask),
    512,
    activation_fn=tf.nn.relu
)

fc2 = slim.fully_connected(
    fc1,
    1024,
    activation_fn=tf.nn.relu
)

reconstructed_image = slim.fully_connected(
    fc2,
    28 * 28,
    activation_fn=tf.nn.sigmoid
)

reconstructed_image = tf.reshape(
    reconstructed_image,
    shape=[-1, 28, 28, 1]
)


loss = tf.reduce_sum(
        target_onehots * tf.math.square(tf.math.maximum(0.0, .9 - digicap_norm)) +
        (1.0 - target_onehots) * y * tf.math.square(tf.math.maximum(0.0, digicap_norm - .1))
    ) + \
    .0005 * tf.reduce_sum(tf.square(reconstructed_image - image))
trainer = tf.train.AdamOptimizer(.0001)
minimize = trainer.minimize(loss)

batch_size = 512
num_epochs = 5
indexes = np.random.permutation(x_train.shape[0])
num_splits = np.ceil(x_train.shape[0] * 1.0 / batch_size)

x_batches = np.array_split(
    x_train[indexes, :],
    num_splits
)

y_batches = np.array_split(
    y_train[indexes],
    num_splits
)

epoch_losses = []
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_count, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            digicap_masks = np.zeros(
                [x_batch.shape[0], num_digits, digicap_dims],
                dtype="float32"
            )
            digicap_masks[range(len(y_batch)), y_batch, :] = 1.0
            _, losses = sess.run(
                [minimize, loss],
                feed_dict={
                    image: x_batch,
                    target: y_batch,
                    digicap_mask: digicap_masks
                }
            )
            epoch_loss += losses
            if batch_count % 5 == 0:
                print(
                    "epoch: {} batch_count: {} epoch running average loss: {}".format(
                        epoch,
                        batch_count,
                        epoch_loss/batch_count
                    )
                )                
        epoch_losses.append(epoch_loss)
        if epoch % 1 == 0:
            print(
                "epoch: {} average loss: {}".format(
                    epoch,
                    np.mean(epoch_losses[-5:])
                )
            )

        saver.save(sess, "./mnist-capsconv", global_step=epoch)

