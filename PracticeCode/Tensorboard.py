# visualize Tensorflow graph
# Follwing codes show how the early version of tensorboard work


# 看不懂这一章的 tensorboard 工具 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = 'to/log/'
BATCH_SIZE = 100
TRAIN_STEPS = 30000

import win_unicode_console
win_unicode_console.enable()

# 生成变量监控信息
def variable_suammaries(var,name):
	with tf.name_scope('summaries'):
		# tf.histogram_summary 函数记录张量中元素的取值分布
		tf.summary.histogram(name,var)
		
		# 计算变量的平均的值
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean/'+name,mean)
		
		# 计算变量的标准差
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
		tf.summary.scalar('stddev/'+name,stddev)
		
# 生成一层全连接层神经网络
def nn_layer(input_tensor,input_dim,output_dim,layer_name,act=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
			variable_suammaries(weights,layer_name+'/weights')
			
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.constant(0.0,shape=[output_dim]))
			variable_suammaries(biases,layer_name+'/biases')
		
		with tf.name_scope('W_plus_b'):
			preactivate = tf.matmul(input_tensor,weights)+biases
			# 记录神经网络输出节点在经过激活函数之前的分布
			tf.summary.histogram(layer_name+'/pre_activations',preactivate)
			
		activations = act(preactivate,name='activation')
		# 记录神经网络输出节点在经过激活函数之后的分布
		# 因为用的是 ReLU 所以所有值都是大于0的
		tf.summary.histogram(layer_name+'/activations',activations)
		return activations

def main(_):
	mnist = input_data.read_data_sets('data/',one_hot=True)
	with tf.name_scope('input'):
		x = tf.placeholder(tf.float32,[None,784],name='x-input')
		y_ = tf.placeholder(tf.float32,[None,10],name='y-input')
	
	# 将输入向量还原成图片的像素矩阵 
	with tf.name_scope('input_reshape'):
		image_shaped_input = tf.reshape(x,[-1,28,28,1])
		tf.summary.image('input',image_shaped_input,10)
	
	
	hidden1 = nn_layer(x,784,500,'layer1')
	y = nn_layer(hidden1,500,10,'layer2',act=tf.identity)
		
	# 计算交叉熵
	with tf.name_scope('cross_entropy'):
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
		tf.summary.scalar('cross_entropy',cross_entropy)
	
	with tf.name_scope('train'):
		train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
	
	# 计算正确率
	with tf.name_scope('accuracy'):
		with tf.name_scope('correct_predicion'):
			correct_predicion = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		with tf.name_scope('accuracy'):
			accuracy = tf.reduce_mean(tf.cast(correct_predicion,tf.float32))
		tf.summary.scalar('accuracy',accuracy)
		
	# tf.scalar_summary tf.histogram_summary tf.image_summary 等函数不会立即执行
	# 需要通过 sess.run 来明确调用这些函数 
	# TensorFlow 提供了 tf.merge_all_summaries 函数来整理所有的日志生成操作
	merged = tf.summary.merge_all()
		
	
	config = tf.ConfigProto(allow_soft_placement=True)
	#最多占gpu资源的70%
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
	#开始不会给tensorflow全部gpu资源 而是按需增加
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
		
	summary_writer = tf.summary.FileWriter(SUMMARY_DIR,sess.graph)
	sess.run(tf.global_variables_initializer())
		
	for i in range(TRAIN_STEPS):
		xs, ys = mnist.train.next_batch(BATCH_SIZE)
		summary,_ = sess.run([merged,train_step],feed_dict={x:xs,y_:ys})
		if i % 1000 == 0:
			print('current step: ',i);
		summary_writer.add_summary(summary,i)
	summary_writer.close()
	
if __name__ == '__main__':
	tf.app.run()
		
		

