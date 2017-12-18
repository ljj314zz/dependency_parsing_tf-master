# -*- coding:utf-8 -*-
#参数初始化
import tensorflow as tf
import math


def random_uniform_initializer(shape, name, val, trainable=True):#生成均匀分布的随机
    out = tf.get_variable(shape=list(shape), dtype=tf.float32,
                          initializer=tf.random_uniform_initializer(minval=-val, maxval=val, dtype=tf.float32),
                          trainable=trainable, name=name)
    return out


def xavier_initializer(shape, name, trainable=True):#xavier分布的随机
    val = math.sqrt(6. / sum(shape))
    return random_uniform_initializer(shape, name, val, trainable=trainable)


def random_normal_initializer(shape, name, mean=0., stddev=1, trainable=True):#生成标准正态分布的随机数
    return tf.get_variable(shape = list(shape), dtype=tf.float32,
                           initializer=tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=tf.float32),
                           trainable=trainable, name=name)

