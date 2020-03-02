import numpy as np
import tensorflow as tf
import os
from Word2Vec_test import Word2Vec

class TextCNN(object):
    def __init__(self, sequence_length, num_classes, word2vec, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda, dropout_keep_prob):
        with tf.Graph().as_default():
            self.prepare_model(sequence_length=sequence_length, num_classes=num_classes, word2vec=word2vec,
                               embedding_size=embedding_size, filter_sizes=filter_sizes, num_filters=num_filters,
                               l2_reg_lambda=l2_reg_lambda, dropout_keep_prob=dropout_keep_prob)
            self.prepare_session()

    def prepare_model(self,sequence_length, num_classes, word2vec, embedding_size, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda=0.0):
        with tf.name_scope("CNN_input"):
            # input,  dropout
            input = tf.placeholder(tf.int32, [None, sequence_length], name='input')
            label = tf.placeholder(tf.int32, [None, 1], name='label')

            label_one_hot = tf.one_hot(label, num_classes)
            label_one_hot = tf.reshape(label_one_hot, [-1, num_classes])
            #dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

            l2_loss = tf.constant(0.0)

        with tf.name_scope('CNN_Embedding'):
            """
            W = tf.Variable(
                tf.random_uniform([12000, embedding_size], -1.0, 1.0),
                name="W")
            embedded_chars = tf.nn.embedding_lookup(W, input)
            embedded_chars = tf.expand_dims(embedded_chars, -1)

            """
            #만든 Word2Vec을 사용하는 코드
            # [None, sequence_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(word2vec, input)
            # [None, sequence_length, embedding_size, 1]
            embedded_chars = tf.expand_dims(embedded_chars, -1)


        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # convolution
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    embedded_chars,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1,  1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)
        #
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # dropout
        with tf.name_scope('dropout'):
            h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

        # prediction
        with tf.name_scope('output'):
            W = tf.get_variable('W', shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            #train용 -drop out 적용
            scores = tf.nn.xw_plus_b(h_drop, W, b, name='scores')

            #Accuracy 계산용 -drop out 미적용
            scores4Acc = tf.nn.xw_plus_b(h_pool_flat, W, b, name='scores4acc')
            #predictions = tf.argmax(scores, 1, name='predictions')
            predictions = tf.argmax(scores4Acc, 1, name='predictions')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=label_one_hot)
            #loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            loss = tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(predictions, tf.argmax(label_one_hot, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        with tf.name_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer(1e-3)
            optimize = optimizer.minimize(loss)

        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)

        # variables
        self.input = input
        self.label = label
        self.dropout_keep_prob = dropout_keep_prob


        self.h_pool = h_pool
        self.h_pool_flat = h_pool_flat
        self.test=pooled_outputs
        self.num_filters_total = num_filters_total
        self.h_drop = h_drop
        self.one_hot = label_one_hot

        self.predictions = predictions
        self.loss = loss
        self.scores = scores
        self.scores4Acc = scores4Acc
        self.accuracy = accuracy
        self.optimize = optimize

        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)

    def prepare_session(self):
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./TextCNNLOG/TextCNNSummary", sess.graph)

        self.saver = saver
        self.sess = sess
        self.summary = summary
        self.writer = writer

if __name__ == '__main__':
    CNN_embedding_size = 128  # Dimension of the embedding vector.
    CNN_filter_sizes = 3, 4, 5,6
    CNN_num_filters = 100
    CNN_dropout_keep_prob = 0.5
    CNN_batch_size = 64
    CNN_sequence_length = 100
    CNN_num_classes = 5
    CNN_l2_reg_lambda = 0.1

    vocabulary_size = 12000

    Word2Vec_batch_size = 128
    Word2Vec_embedding_size = 128  # Dimension of the embedding vector.
    Word2Vec_skip_window = 4  # How many words to consider left and right.
    Word2Vec_num_skips = 8  # How many times to reuse an input to generate a label.
    Word2Vec_num_sampled = 64  # Number of negative examples to sample.

    Word2Vec_valid_size = 16  # Random set of words to evaluate similarity on.
    Word2Vec_valid_window = 100  # Only pick dev samples in the head of the distribution.
    Word2Vec_valid_examples = np.random.choice(Word2Vec_valid_window, Word2Vec_valid_size, replace=False)

    word2vec = Word2Vec(
        vocabulary_size=vocabulary_size, embedding_size=Word2Vec_embedding_size,
        batch_size=Word2Vec_batch_size, num_sampled=Word2Vec_num_sampled, valid_size=Word2Vec_valid_size,
        valid_window=Word2Vec_valid_window, valid_examples=Word2Vec_valid_examples)

    word2vec_Saver = word2vec.saver

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__) + "/Saver_300000TrainStep")
    if ckpt and ckpt.model_checkpoint_path:
        word2vec_Saver.restore(word2vec.sess, ckpt.model_checkpoint_path)

    textCNN = TextCNN(sequence_length=CNN_sequence_length, num_classes=CNN_num_classes,
                      word2vec=word2vec.normalized_embeddings.eval(), embedding_size=CNN_embedding_size,
                      filter_sizes=CNN_filter_sizes, num_filters=CNN_num_filters, l2_reg_lambda=CNN_l2_reg_lambda,
                      dropout_keep_prob=CNN_dropout_keep_prob)

