import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')

# helper test function
def viz_batch(imgs):                               

    N = imgs.shape[1]

    fig, axes = plt.subplots(1, N, figsize=(20,5))

    for n in range(N):
        axes[n].imshow(imgs[0,n,:784].reshape(28,28))
        axes[n].set_title(f'Label:{imgs[0,n,-2:]}')
    plt.show()


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """

    labels = tf.reshape(labels, shape=(tf.shape(labels)[0], labels.shape[1]*labels.shape[2],
                                       labels.shape[3]))[:, -int(preds.shape[-1]):, :]

    preds = preds[:, -int(preds.shape[-1]):, :]

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=preds))

    return loss

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """

        support_batch = tf.concat([input_images[:,:-1,:,:], input_labels[:,:-1,:,:]], axis=3)
        pred_batch = tf.concat([input_images[:, -1:, :, :], tf.zeros(shape=(tf.shape(input_labels)[0], 1,
                                                                           self.num_classes, self.num_classes))],
                                                                           axis=3)

        inp_batch = tf.concat([support_batch, pred_batch], axis=1)

        inp_batch = tf.reshape(inp_batch, shape=(tf.shape(input_images)[0], self.num_classes * self.samples_per_class,
                                                 input_images.shape[-1] + self.num_classes ))

        out = self.layer2(self.layer1(inp_batch))

        return out

ims = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, 784))

labels = tf.placeholder(tf.float32, shape=(
    None, FLAGS.num_samples + 1, FLAGS.num_classes, FLAGS.num_classes))

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)

o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
out = o(ims, labels)

loss = loss_function(out, labels)
optim = tf.train.AdamOptimizer(0.001)
optimizer_step = optim.minimize(loss)

# Setup TensorBoard logging
train_log_dir = 'logs/problem_3/' + f'K_{FLAGS.num_samples}_N_{FLAGS.num_classes}' + '/train'
test_log_dir = 'logs/problem_3/' + f'K_{FLAGS.num_samples}_N_{FLAGS.num_classes}' + '/test'

with tf.Session() as sess:

    train_summary_writer = tf.summary.FileWriter(train_log_dir)
    test_summary_writer = tf.summary.FileWriter(test_log_dir)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    for step in range(50000):
        i, l = data_generator.sample_batch('train', FLAGS.meta_batch_size)
        feed = {ims: i.astype(np.float32), labels: l.astype(np.float32)}

        _, ls = sess.run([optimizer_step, loss], feed)

        if step % 100 == 0:
            print("*" * 5 + "Iter " + str(step) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            feed = {ims: i.astype(np.float32),
                    labels: l.astype(np.float32)}
            pred, tls = sess.run([out, loss], feed)
            print("Train Loss:", ls, "Test Loss:", tls)
            pred = pred.reshape(
                -1, FLAGS.num_samples + 1,
                FLAGS.num_classes, FLAGS.num_classes)
            pred = pred[:, -1, :, :].argmax(2)
            l = l[:, -1, :, :].argmax(2)
            test_acc = (1.0 * (pred == l)).mean()
            print(f"Test Accuracy: {test_acc}")

            # writing summaries
            test_loss_sum = tf.Summary()
            test_loss_sum.value.add(tag='test_loss', simple_value=tls)
            test_summary_writer.add_summary(test_loss_sum, step)

            train_loss_sum = tf.Summary()
            train_loss_sum.value.add(tag='train_loss', simple_value=ls)
            train_summary_writer.add_summary(train_loss_sum, step)

            test_acc_sum = tf.Summary()
            test_acc_sum.value.add(tag='test_acc', simple_value=test_acc)
            test_summary_writer.add_summary(test_acc_sum, step)

    train_summary_writer.flush()
    train_summary_writer.close()
    test_summary_writer.flush()
    test_summary_writer.close()





















