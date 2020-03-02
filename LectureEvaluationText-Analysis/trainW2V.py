# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
from tempfile import gettempdir
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import pylab as pl

from tensorflow.contrib.tensorboard.plugins import projector
from Word2Vec_test import Word2Vec
from KMeans import KMeans


# Read the data into a list of strings.
def read_data(dir, filename1, filename2, filename3):
    """

    :param filename:
    :return:
    """

    data = list()
    """Extract the first file enclosed in a zip file as a list of words."""
    filename = os.path.join(dir, filename1)
    with open(filename, 'r', encoding='UTF8') as f:
        data.extend(tf.compat.as_str(f.read()).split())  # compat.as_str() <- string으로 encoding 함수
        print(len(data))

    filename = os.path.join(dir, filename2)
    with open(filename, 'r', encoding='UTF8') as f:
        data.extend(tf.compat.as_str(f.read()).split())  # compat.as_str() <- string으로 encoding 함수
        print(len(data))

    filename = os.path.join(dir, filename3)
    with open(filename, 'r', encoding='UTF8') as f:
        data.extend(tf.compat.as_str(f.read()).split())  # compat.as_str() <- string으로 encoding 함수
        print(len(data))

    return data


def build_dataset(words, n_words):
    """

    :param words: All splited words from 'text8.zip'
    :param n_words: Size of words that use
    :return: count: Most_common 'n_words' size words in 'words'
              dictionary: (word : index) dictionary
              reversed_dictionary : (index : word) dictionary, literaly reverse the dictionary
    """
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]

    """
    count.extend(collections.Counter(words).most_common(
        n_words - 1))  # Counter는 리스트내의 요소들이 몇번이나 반복되어 나타났는지를 counting하여 반환, most_common(n)는 상위 n개를 리턴
    print(count)
    """
    print(len(collections.Counter(words)))


    dictionary = dict()
    for i, (word, _) in enumerate(count):
        dictionary[word] = i
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)  # dictionary에 word가 존재하지 않는 경우 0을 반환
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count  # unknown인 단어 수 저장
    reversed_dictionary = dict(
        zip(dictionary.values(), dictionary.keys()))  # (index : word)로 구성된 dictionary ->나중에 탐색하기 편하게 하기위함

    with open("Data\\IndexDictionary.txt", 'w', encoding='utf-8') as f:
        for i in range(len(reversed_dictionary)):
            f.write(str(i) + "\t"+ reversed_dictionary[i]+"\n" )


    return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    """

    :param batch_size:  SGD 알고리즘에 적용할 데이터 갯수. 한 번에 처리할 크기
    :param num_skips: context window에서 구축할 (target, context) 쌍의 갯수, num_skips는 보통 경우 skip_window*2
    :param skip_window: skip-gram 모델에 사용할 윈도우 크기
    :return:
    """
    global data_index
    assert batch_size % num_skips == 0  # batch_size % num_skips == 0이 아니면 error를 내줘! 라는 의미
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # span size만큼의 queue를 항상 유지(최신 데이터로)
    if data_index + span > len(data):  # data를 한바뀌 다 돈 경우
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span

    # batch 와 label에 각각 target, context를 넣는 작업
    for i in range(batch_size // num_skips):  # //은 몫 연산자
        context_words = [w for w in range(span) if w != skip_window]  # skip_window에는 target이 존재함으로
        random.shuffle(context_words)  # index들을 random으로 형성
        words_to_use = collections.deque(context_words)
        for j in range(num_skips):
            batch[i * num_skips + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            # data의 끝까지 한번 돈 경우
            print("끝까지 옴")
            buffer = collections.deque(maxlen=span)
            buffer.extend(data[0:span])
            data_index = span
        else:
            # 현재 skip_window 인덱스 위치에 있던 target으로 context생성이 끝났으면 다음 데이터를 buffer에 넣어 target을 갱신
            buffer.append(data[data_index])
            data_index += 1

    # Backtrack a little bit to avoid skipping words in the end of a batch
    # data_index는 반복문에서 batch_size // num_skips 만큼 증가한다.
    # 최정적인 data_index는 마지막을 지난 위치를 가리키게 되니까, 정확한 계산을 위해 앞으로 돌려놓을 필요가 있다.
    # span에 context 윈도우 전체 크기가 있으니까, span만큼 뒤로 이동한다.
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

    # pylint: disable=missing-docstring
    # Function to draw visualization of distance between embeddings.


def plot_with_labels(low_dim_embs, labels, filename):
    """

    :param low_dim_embs:
    :param labels:
    :param filename:
    :return:
    """
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    dir = os.getcwd()
    filePath = os.path.join(dir, "Data")
    vocabulary = read_data(dir=filePath, filename1="serial1.txt", filename2="serial2.txt", filename3="serial3.txt")

    # Step 1: Download the data.




    # Step 2: Build the dictionary and replace rare words with UNK token.
    vocabulary_size = 12000
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)

    del vocabulary

    # Step 3: Function to generate a training batch for the skip-gram model. = generate_batch(batch_size, num_skips, skip_window)
    data_index = 0

    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 4  # How many words to consider left and right.
    num_skips = 8  # How many times to reuse an input to generate a label.
    num_sampled = 64  # Number of negative examples to sample.

    # We pick a random validation set to sample nearest neighbors. Here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent. These 3 variables are used only for
    # displaying model accuracy, they don't affect calculation.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)

    # Step 4: Build and train a skip-gram model.
    word2vec = Word2Vec(
        vocabulary_size=vocabulary_size, embedding_size=embedding_size,
        batch_size=batch_size, num_sampled=num_sampled, valid_size=valid_size,
        valid_window=valid_window, valid_examples=valid_examples)

    PATH = os.getcwd()
    LOG_DIR = os.path.join(PATH, "Saver_300000TrainStep")

    #
    """
    
    saver = word2vec.saver
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__) + "/Saver_200000TrainStep")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(word2vec.sess, ckpt.model_checkpoint_path)
        print(word2vec.embeddings.eval())
"""
    #

    # Step 5: Begin training.
    num_steps = 300001
    average_loss = 0

    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)
        feed_dict = {word2vec.train_inputs: batch_inputs, word2vec.train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = word2vec.sess.run([word2vec.optimizer, word2vec.loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            summary, loss_val = word2vec.sess.run([word2vec.summmary, word2vec.loss], feed_dict=feed_dict)
            word2vec.writer.add_summary(summary, global_step=step)
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = word2vec.similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)

    final_embeddings = word2vec.normalized_embeddings.eval()
    word2vec.saver.save(word2vec.sess, os.path.join(LOG_DIR, 'images'))
    embeddings = word2vec.embeddings.eval()
    print("Word2Vec Train Complete")

    # Step 6 : Begin Dimensions reduction


    # Step 7 : Begin K-Means Clustring
    classes = 5
    clusted_embeddings = KMeans(classes=classes, low_dim_embeddings=embeddings)
    print("Clustring Complete")

    # Step 8 : Visualize the embeddings with TensorBoard
    metadata = os.path.join(LOG_DIR, 'metadata.tsv')
    with open(metadata, 'w', encoding='UTF8') as metadata_file:
        metadata_file.write("Index" + '\t' + "Word" + '\t' + "Cluster" + '\n')
        for i, data in enumerate(reverse_dictionary.values()):
            metadata_file.write(str(i) + '\t' + data + '\t' + str(clusted_embeddings[i]['cluster_idx']) + '\n')

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = word2vec.embeddings.name

    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    summary_writer = tf.summary.FileWriter(LOG_DIR)

    projector.visualize_embeddings(summary_writer, config)

    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from matplotlib import font_manager, rc

        font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
        rc('font', family=font_name)

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(), 'tsne.png'))

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)
