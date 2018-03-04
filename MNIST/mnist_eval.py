# coding:utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载 mnist_inference.py 和 mnist_train.py 
import mnist_inference
import mnist_train

INPUT_NODE = mnist_inference.INPUT_NODE
OUTPUT_NODE = mnist_inference.OUTPUT_NODE
LAYER1_NODE = mnist_inference.LAYER1_NODE


# 解决CMD端的OSError
import win_unicode_console
win_unicode_console.enable()

def evalute(mnist):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
		y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
		
		validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
		
		# 测试的时候不关注正则化损失，因此参数设置为 None
		y = mnist_inference.inference(x,None)
		
		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
		
		variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
		
		# 限定任务占用的GPU内存量
		config = tf.ConfigProto(allow_soft_placement=True)
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
				print('After %s training step(s), validation accuracy = %g'%(global_step,accuracy_score))
			else:
				print('No checkpoint file found')
				return 
			

def main(argv=None):
	mnist = input_data.read_data_sets('data/',one_hot=True)
	evalute(mnist)
	
if __name__ == '__main__':
	tf.app.run()
