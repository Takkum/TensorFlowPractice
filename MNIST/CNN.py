# 用卷积神经网络来识别 MNIST
# 在跑了15个epoch下，识别率大概在99%
 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集的一些常数
INPUT_NODE = 784	# 输入层的节点数
OUTPUT_NODE = 10	# 输出层的节点数 这里是0-9这10个输出数字

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 3
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 3
# 全连接层节点个数
FC_SIZE = 256

# 配置神经网络参数
LEARNING_RATE = 0.001
BATCH_SIZE = 100

def train(mnist):
	# 输入是一个四维矩阵。
	# 第一维是样例个数，第二、三维是图片的尺寸，第四维是图片深度
	# 采用了 dropout 方法，用到了 keep_prob
	X = tf.placeholder(tf.float32,[None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
	Y = tf.placeholder(tf.float32,[None,NUM_LABELS])
	keep_prob = tf.placeholder(tf.float32)
	
	# 第一层是卷积层，定义前向传播过程。
	# 输入矩阵：?*28*28*1
	# 输出矩阵：?*28*28*32
	W1 = tf.Variable(tf.random_normal([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],stddev=0.01))
	L1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
	L1 = tf.nn.relu(L1)
	
	# 第二层是下采样层
	# 输入矩阵：?*28*28*32
	# 输出矩阵：?*14*14*32
	L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
	# 第三层是卷积层
	# 输入矩阵：?*14*14*32
	# 输出矩阵：?*14*14*64
	W2 = tf.Variable(tf.random_normal([CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],stddev=0.01))
	L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
	L2 = tf.nn.relu(L2)
	
	# 第四层是下采样层
	# 输入矩阵：?*14*14*64
	# 输出矩阵：?*7*7*64
	L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
	# 第五层是全连接层。将 L2(?*7*7*64) 矩阵 拉成 ?*3136 的矩阵
	# dropout 一般只在全连接层而不是卷积层或者下采样层使用
	# 输入矩阵：?*7*7*64
	# 输出矩阵：?*256
	pool_shape = L2.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	
	W3 = tf.Variable(tf.random_normal([nodes,FC_SIZE],stddev=0.01))
	# 关于 tf.reshape 函数中shape参数里 -1 的含义请参考官方文档
	# 这里-1 代表了?
	L3 = tf.reshape(L2,[-1,nodes])
	L3 = tf.nn.relu(tf.matmul(L3,W3))
	L3 = tf.nn.dropout(L3,keep_prob)
	
	# 第六层是全连接层
	# 输入矩阵：?*256
	# 输出矩阵：?*10
	W4 = tf.Variable(tf.random_normal([FC_SIZE,NUM_LABELS],stddev=0.01))
	logit = tf.matmul(L3,W4)
	
	# 最后通过 softmax 层输出结果
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=Y))
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
	
	# 限定任务占用 GPU 内存量
	config = tf.ConfigProto(allow_soft_placement=True)
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	total_batch = int(mnist.train.num_examples / BATCH_SIZE)
	
	# 跑15个epoch 
	for epoch in range(15):
		total_cost = 0
		for i in range(total_batch):
			batch_xs,batch_ys = mnist.train.next_batch(BATCH_SIZE)
			# batch_xs.shape 可以查看 array 的维度 100*784
			# 把训练数据格式调整为一个四维矩阵
			batch_xs = batch_xs.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)
			
			_,cost_val = sess.run([optimizer,cost],
								  feed_dict={X:batch_xs,
											 Y:batch_ys,
											 keep_prob:0.7})
			total_cost += cost_val
			
		print('Epoch:','%04d'% (epoch+1),'Average cost =','{:3f}'.format(total_cost/total_batch))
		
	print('All the epoches are over!')
						
	# 在 Validation 上验证正确率
	# 在测试集上不采用 dropout 方法， keep_prob 设置为1
	is_correct = tf.equal(tf.argmax(logit,1),tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
	print('the accuracy is ',sess.run(accuracy*100,feed_dict={
												X:mnist.validation.images.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS),
												Y:mnist.validation.labels,
												keep_prob:1}),'%')

# 主程序
def main(argv=None):
	mnist = input_data.read_data_sets("data/",one_hot=True)
	train(mnist)
	
if __name__ == '__main__':
	tf.app.run()

