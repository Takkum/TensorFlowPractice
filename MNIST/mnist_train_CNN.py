# coding:utf-8
import os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 加载 mnist_inference_CNN.py
import mnist_inference_CNN
INPUT_NODE = mnist_inference_CNN.INPUT_NODE
OUTPUT_NODE = mnist_inference_CNN.OUTPUT_NODE
IMAGE_SIZE = mnist_inference_CNN.IMAGE_SIZE
NUM_CHANNELS = mnist_inference_CNN.NUM_CHANNELS

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1	# 之前是0.8
LEARNING_RATE_DECAY = 0.99	
REGULARAZION_RATE = 0.0001
TRAINING_STEPS = 55000	# 之前是30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'

# 解决CMD端的OSError
import win_unicode_console
win_unicode_console.enable()

def train(mnist):
	# x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
	# 输入是一个四维矩阵
	# 第一维表示一个 batch 中样例的个数
	# 第二维和第三维都表示图片的尺寸 28*28
	# 第四维表示图片的深度，对于RGB格式的图片，深度为3.
	x = tf.placeholder(tf.float32,[
						BATCH_SIZE,				
						IMAGE_SIZE,
						IMAGE_SIZE,
						NUM_CHANNELS],name='x-input')
	y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZION_RATE)
	y = mnist_inference_CNN.inference(x,True,regularizer)
	global_step = tf.Variable(0,trainable=False)
	
	# 定义损失函数、学习率、滑动平均操作以及训练过程
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
	
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	
	with tf.control_dependencies([train_step,variable_averages_op]):
		train_op = tf.no_op(name='train')
	
	
	# 限定任务占用的GPU内存量
	config = tf.ConfigProto(allow_soft_placement=True)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	
	# 初始化 Tensorflow 持久化类
	saver = tf.train.Saver()
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		
		for i in range(TRAINING_STEPS):
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			reshaped_xs = np.reshape(xs,( BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
			_,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
			
			if i % 1000 == 0:
				print('After %d training step(s), loss on training batch is %g'%(step,loss_value))
				
				# 保存训练1000轮的模型。
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

				
def main(argv=None):
	mnist = input_data.read_data_sets('data/',one_hot=True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()