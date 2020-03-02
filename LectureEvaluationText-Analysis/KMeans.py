import tensorflow as tf
import numpy as np
from tensorflow.contrib.factorization.python.ops import clustering_ops


def KMeans(classes, low_dim_embeddings):
    """

    :param classes: number of Clusters
    :param low_dim_embeddings: Dimensionn reduced embedding word vectors
    :return:
    """
    row = len(low_dim_embeddings)
    col = len(low_dim_embeddings[0])

    print("[", row, "x", col, "] sized input")

    def train_input_fn():
        data = tf.constant(low_dim_embeddings, tf.float32)
        return (data, None)

    def predict_input_fn():
        return np.array(low_dim_embeddings, np.float32)

    model = tf.contrib.learn.KMeansClustering(
        classes
        , distance_metric=clustering_ops.SQUARED_EUCLIDEAN_DISTANCE  # SQUARED_EUCLIDEAN_DISTANCE, COSINE_DISTANCE
        , initial_clusters=tf.contrib.learn.KMeansClustering.RANDOM_INIT
    )
    model.fit(input_fn=train_input_fn, steps=10000)

    predictions = model.predict(input_fn=predict_input_fn, as_iterable=True)

    predictict_Result = []
    i = 0
    for index in predictions:
        #print("[ Word2Vec ", i, "index] -> cluster_", index['cluster_idx'])
        i = i + 1
        predictict_Result.append(index)

    return predictict_Result