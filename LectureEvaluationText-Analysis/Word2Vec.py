import tensorflow as tf
import numpy as np
import math


class Word2Vec:
    def __init__(self, vocabulary_size, embedding_size, batch_size, num_sampled,valid_size, valid_window, valid_examples):
        with tf.Graph().as_default():
            self.prepare_model(
                vocabulary_size=vocabulary_size,
                embedding_size=embedding_size,
                batch_size=batch_size,
                num_sampled=num_sampled, valid_size= valid_size, valid_window= valid_window, valid_examples = valid_examples)
            self.prepare_session()

    def prepare_model(self, vocabulary_size, embedding_size, batch_size, num_sampled, valid_size, valid_window, valid_examples):
        """

        :param vocabulary_size:
        :param embedding_size:
        :param batch_size:
        :param num_sampled:
        :return:
        """


        with tf.name_scope("Word2Vec_input"):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.name_scope("embedding"):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="word2vec")
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        with tf.name_scope("optimizer"):
            # Compute the average NCE loss for the batch.
            # tf.nce_loss automatically draws a new sample of the negative labels each
            # time we evaluate the loss.
            # Explanation of the meaning of NCE loss:
            #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))

            # Construct the SGD optimizer using a learning rate of 1.0.
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        with  tf.name_scope("similarity"):
            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = tf.divide(embeddings, norm, name="normalized_embeddings")

            valid_embeddings = tf.nn.embedding_lookup(
                normalized_embeddings, valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)


        tf.summary.scalar("loss", loss)
        tf.summary.histogram("word2vec", embeddings)
        tf.summary.histogram("nce_weights", nce_weights)
        tf.summary.histogram("nce_biases", nce_biases)

        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.optimizer = optimizer
        self.loss = loss
        self.similarity = similarity
        self.normalized_embeddings = normalized_embeddings
        self.embeddings = embeddings



    def prepare_session(self):
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(".\\Summary", sess.graph)

        self.saver = saver
        self.sess = sess
        self.summmary = summary
        self.writer = writer


