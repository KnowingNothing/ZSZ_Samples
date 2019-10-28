from __future__ import absolute_import

import functools
import time
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf 
from tensorflow.contrib.compiler import xla


def gemm1(A, B):
    return tf.linalg.matmul(A, B)


def gemm2(A, B):
    return tf.reduce_sum(
        tf.multiply(
            tf.tile(tf.expand_dims(A, -1), [1, 1, B.shape[1]]), 
            tf.tile(tf.expand_dims(B, 0), [A.shape[0], 1, 1])
        ),
        axis=1
    )


def gemm3(A, B):
    tmp_ary = []
    for i in range(A.shape[0]):
        tmp_row = []
        for j in range(B.shape[1]):
            tmp = A[i, 0] * B[0, j]
            for k in range(1, A.shape[1]):
                tmp = tmp + A[i, k] * B[k, j]
            tmp_row.append(tmp)
        tmp_ary.append(tf.stack(tmp_row))

    return tf.stack(tmp_ary)


def run(gemm, M, N, K, repeat=10):
    A = tf.placeholder(name="A", dtype=tf.float32, shape=(M, K))
    B = tf.placeholder(name="B", dtype=tf.float32, shape=(K, N))
    create_graph = functools.partial(gemm)
    [C] = xla.compile(create_graph, inputs=[A, B])
    # C = create_graph(A, B)
    A_np = np.random.uniform(0, 1, (M, K)).astype(np.float32)
    B_np = np.random.uniform(0, 1, (N, K)).astype(np.float32)

    # A_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # B_np = np.array([[5, 6], [7, 8]], dtype=np.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # warm-up
        C_np = sess.run(C, feed_dict={A: A_np, B: B_np})
        beg = time.time()
        for i in range(repeat):
            C_np = sess.run(C, feed_dict={A: A_np, B: B_np})
        end = time.time()

    return (end - beg) * 1e3 / repeat


if __name__ == "__main__":
    print("gemm1:")
    for i in range(14):
        M = 2 ** i 
        time_cost = run(gemm1, M, M, M)
        print("     (%d x %d): %fms" % (M, M, time_cost))
    print()
    print("gemm2:")
    for i in range(14):
        M = 2 ** i 
        time_cost = run(gemm2, M, M, M)
        print("     (%d x %d): %fms" % (M, M, time_cost))
    
