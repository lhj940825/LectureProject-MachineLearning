import tensorflow as tf
from Word2Vec_test import Word2Vec
import numpy as np
import os
# from TextCNN import TextCNN
from TextCNN import TextCNN
import csv

"""======================================================================================================================"""
# File Directory Variable

INDEXED_DICTIONARY_FILENAME = "Data\\IndexDictionary.txt"
INDEXED_TOKENIZED_ASSESSMENT_RESULT_FILENAME = "Data\\IndexedTokenizedAssessmentResult.txt"

"""======================================================================================================================"""
# Word2Vec Variable

vocabulary_size = 12000

Word2Vec_batch_size = 128
Word2Vec_embedding_size = 128  # Dimension of the embedding vector.
Word2Vec_skip_window = 4  # How many words to consider left and right.
Word2Vec_num_skips = 8  # How many times to reuse an input to generate a label.
Word2Vec_num_sampled = 64  # Number of negative examples to sample.

Word2Vec_valid_size = 16  # Random set of words to evaluate similarity on.
Word2Vec_valid_window = 100  # Only pick dev samples in the head of the distribution.
Word2Vec_valid_examples = np.random.choice(Word2Vec_valid_window, Word2Vec_valid_size, replace=False)
"""======================================================================================================================"""
# TextCNN Variable

CNN_embedding_size = 128  # Dimension of the embedding vector.
CNN_filter_sizes = 3, 4, 5, 6
CNN_num_filters = 100
CNN_dropout_keep_prob = 0.5
CNN_batch_size = 60
CNN_sequence_length = 30
CNN_num_classes = 2
CNN_l2_reg_lambda = 0.1

"""======================================================================================================================"""
# Global Variable

data_index = 0
# num_steps = 2500
num_steps = 601
average_loss = 0
PATH = os.getcwd()
LOG_DIR = PATH + "\\TextCNNLOG\\Saver\\"
K_Size = 10
"""======================================================================================================================"""


def read_data(filename):
    xy = np.loadtxt(filename, delimiter=',', dtype=np.int32)
    xy = np.concatenate((xy[:, 0:30], xy[:, [-1]]), axis=1)
    print(np.shape(xy))

    label_data = xy[:, -1]

    positive_data = list()
    negative_data = list()
    total_data = list()

    #긍정 라벨 데이터, 부정 라벨 데이터 각각 2330개
    each_data_size = 2330

    for i in range(len(label_data)):

        if (label_data[i] == 1 and len(negative_data) != each_data_size):
            negative_data.append(xy[i])
        elif (label_data[i] == 2 and len(positive_data) != each_data_size):
            positive_data.append(xy[i])

    for i in range(each_data_size):
        total_data.append(negative_data[i])
        total_data.append(positive_data[i])

    total_data = np.asanyarray(total_data)

    x_data = total_data[:, 0:-1]
    y_data = total_data[:, -1]
    y_data = y_data - 1  # label이 1~5 임으로 0~4로 변경

    return x_data, y_data


def generate_cnn_batch(batch_size, x_data, y_data):
    global data_index
    if data_index + batch_size > len(x_data):
        data_index = 0

    x_batch = x_data[data_index:data_index + batch_size, :]
    y_batch = y_data[data_index:data_index + batch_size]
    y_batch = np.reshape(y_batch, [batch_size, -1])

    data_index += batch_size
    return x_batch, y_batch


def generate_data_for_CV_fold(filename, step):
    data_x, data_y = read_data(filename)

    data_y = np.reshape(data_y, [-1, 1])

    print("index", int(len(data_y) / K_Size) * step, int(len(data_y) / K_Size) * (step + 1))

    test_x = data_x[int(len(data_y) / K_Size) * step: int(len(data_y) / K_Size) * (step + 1):]
    test_y = data_y[int(len(data_y) / K_Size) * step: int(len(data_y) / K_Size) * (step + 1):]

    data_x = list(data_x)
    data_y = list(data_y)

    del data_x[int(len(data_y) / K_Size) * step: int(len(data_y) / K_Size) * (step + 1):]
    del data_y[int(len(data_y) / K_Size) * step: int(len(data_y) / K_Size) * (step + 1):]

    train_x = np.array(data_x)

    train_y = np.array(data_y)

    return train_x, train_y, test_x, test_y


"""
def CV_fold_Accuracy(K_Size, filename, textCNN):
    test_x, test_y = read_data(filename=filename)
    test_y = np.reshape(test_y, [-1, 1])

    average_accuracy = 0

    for step in range(K_Size):
        x_batch = test_x[int(len(test_y) / K_Size) * step \
            :int(len(test_y) / K_Size) * step + int(len(test_y) / K_Size), :]
        y_batch = test_y[int(len(test_y) / K_Size) * step \
            :int(len(test_y) / K_Size) * step + int(len(test_y) / K_Size), :]
        feed_dict = {textCNN.input: test_x, textCNN.label: test_y}
        accuracy = textCNN.sess.run(textCNN.accuracy, feed_dict=feed_dict)
        average_accuracy += accuracy

    average_accuracy /= K_Size
    print("CV_fold Accuracy:", average_accuracy)
"""

if __name__ == "__main__":

    ###

    # Variables for 10-CV
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    ###


    cv_fold_average = 0

    for i in range(K_Size):

        train_x, train_y, test_x, test_y = generate_data_for_CV_fold(
            filename=INDEXED_TOKENIZED_ASSESSMENT_RESULT_FILENAME, step=i)

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

        # Train using Train Dataset
        for step in range(num_steps):
            x_batch, y_batch = generate_cnn_batch(CNN_batch_size, x_data=train_x, y_data=train_y)
            feed_dict = {textCNN.input: x_batch, textCNN.label: y_batch}

            _, loss_val = textCNN.sess.run([textCNN.optimize, textCNN.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 50 == 0:
                summary, loss_val, acc = textCNN.sess.run([textCNN.summary, textCNN.loss, textCNN.accuracy],
                                                          feed_dict=feed_dict)
                textCNN.writer.add_summary(summary, global_step=step)
                if step > 0:
                    average_loss /= 50

                print('Average loss at step', step, ': ', average_loss, "Train Accuracy: ", acc)
                average_loss = 0

        # Calculate Accuracy using Test Dataset
        feed_dict = {textCNN.input: test_x, textCNN.label: test_y}
        accuracy = textCNN.sess.run(textCNN.accuracy, feed_dict=feed_dict)
        print("Accuracy", step, ': ', accuracy)

        # Calculate Confusion Matrix
        feed_dict = {textCNN.input: test_x, textCNN.label: test_y}
        predict = textCNN.sess.run(textCNN.predictions, feed_dict=feed_dict)

        with open('cnn.csv', 'a', encoding='utf-8', newline="")as f:
            wr = csv.writer(f)
            wr.writerow(np.squeeze(predict))
            wr.writerow(np.squeeze(test_y))

        for step in range(len(predict)):

            if test_y[step] == [0] and predict[step] == [0]:
                FN += 1
            if test_y[step] == [1] and predict[step] == [0]:
                TN += 1
            if test_y[step] == [1] and predict[step] == [1]:
                TP += 1
            if test_y[step] == [0] and predict[step] == [1]:
                FP += 1

        print("pp pn np nn", TP, TN, FP, FN)
        ###



        cv_fold_average += accuracy

    textCNN.saver.save(textCNN.sess, os.path.join(LOG_DIR, 'images'))

    # CV_fold_Accuracy(K_Size=K_Size, filename=INDEXED_TOKENIZED_ASSESSMENT_TEST_RESULT_FILENAME, textCNN=textCNN)
    cv_fold_average /= K_Size
    print("cv_fold_accuracy", cv_fold_average)









    # num_filters_total = num_filters * len(filter_sizes)
"""
    testing = tf.nn.embedding_lookup(word2vec.embeddings.eval(), [[0,1,2],[1,3,4],[3,5,6],[5,7,8],[6,9,6]])
    testing_expand = tf.expand_dims(testing, -1)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        _ = sess.run(testing)
        print(np.shape(_))
        _ = sess.run(testing_expand)
        print(np.shape(_))

"""
