from __future__ import absolute_import

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import numpy as np
from tensorflow.python.client import timeline


def depthwise_seperable_block(name, inputs, pointwise_channel, downsample=False):
    stride = 2 if downsample else 1

    depthwise = slim.separable_convolution2d(inputs, num_outputs=None, stride=stride, depth_multiplier=1, kernel_size=[3, 3], scope="%s/depthwise" % name)
    bn = slim.batch_norm(depthwise, scope="%s/dw_batch_norm" % name)
    pointwise = slim.convolution2d(bn, pointwise_channel, kernel_size=[1, 1], scope="%s/pointwise" % name)
    bn = slim.batch_norm(pointwise, scope="%s/pw_batch_norm" % name)
    return bn


def mobilenet(name, inputs, num_class=2, width_mult=1.0, train=False):
    block = depthwise_seperable_block
    with tf.variable_scope(name):
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d], activation_fn=None, 
                        weights_initializer=slim.initializers.xavier_initializer()):
            with slim.arg_scope([slim.batch_norm], is_training=train, activation_fn=tf.nn.relu, fused=True):
                net = slim.convolution2d(inputs, round(32 * width_mult), [3, 3], stride=2, padding="SAME", scope="first_conv3x3")
                net = slim.batch_norm(net, scope="first_batch_norm")
                net = block("depthwise_seperable_1", net, round(64 * width_mult))
                net = block("depthwise_seperable_2", net, round(128 * width_mult), downsample=True)
                net = block("depthwise_seperable_3", net, round(128 * width_mult))
                net = block("depthwise_seperable_4", net, round(256 * width_mult), downsample=True)
                net = block("depthwise_seperable_5", net, round(256 * width_mult))
                net = block("depthwise_seperable_6", net, round(512 * width_mult), downsample=True)
                net = block("depthwise_seperable_7", net, round(512 * width_mult))
                net = block("depthwise_seperable_8", net, round(512 * width_mult))
                net = block("depthwise_seperable_9", net, round(512 * width_mult))
                net = block("depthwise_seperable_10", net, round(512 * width_mult))
                net = block("depthwise_seperable_11", net, round(512 * width_mult))
                net = block("depthwise_seperable_12", net, round(1024 * width_mult), downsample=True)
                net = block("depthwise_seperable_13", net, round(1024 * width_mult))
                net = slim.avg_pool2d(net, [7, 7], scope="avg_pool2d")
        net = tf.squeeze(net, [1, 2], name="squeeze")
        logits = slim.fully_connected(net, num_class, activation_fn=None, scope="fc", 
                                weights_initializer=slim.initializers.xavier_initializer())
        predictions = slim.softmax(logits, scope="softmax")

    return predictions


if __name__ == "__main__":
    batch_size = 1
    data_shape = (batch_size, 224, 224, 3)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter("graph", sess.graph)
        inputs = tf.placeholder(name="input", dtype=tf.float32, shape=data_shape)
        preds = mobilenet("mobilenet", inputs)
        
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        sess.run(tf.global_variables_initializer())

        # # warm up, no record
        # output = sess.run(preds, options=options, 
        #                 feed_dict={inputs: np.random.uniform(1e9, 1e10, data_shape).astype(np.float32)})

        # record
        for i in range(2):
            run_metadata = tf.RunMetadata()
            output = sess.run(preds, options=options, 
                            feed_dict={inputs: np.random.uniform(1e9, 1e10, data_shape).astype(np.float32)}, 
                            run_metadata=run_metadata)
            print(output)
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open("mobilenet-v1-timeline_%d.json" % (i+1), "w") as fout:
                fout.write(chrome_trace)
            