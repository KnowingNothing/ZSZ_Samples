from __future__ import absolute_import

import tensorflow as tf 
from tensorflow.contrib.compiler import xla


def gemm(M, N, K):
    A = tf.placeholder(name="A", dtype=tf.float32, shape=(M, K))
    B = tf.placeholder(name="B", dtype=tf.float32, shape=(N, K))
    
