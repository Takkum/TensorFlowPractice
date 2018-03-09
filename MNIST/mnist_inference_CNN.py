# coding:utf-8
import tensorflow as tf


# 配置神经网络的参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512

# 定义卷积神经网络的前向传播过程。这里添加一个新的参数 train，用于区分训练过程和测试过程。
# 程序会用到 dropout 方法，dropout 可以进一步提升模型可靠性并防止过拟合
# dropout 过程只在训练时使用。
def inference(input_tensor,train,regularizer):
	# 声明第一层卷积层的变量并实现前向传播过程。
	# 通过声明命名空间来隔离不同层的变量，能让每一层中的变量命名只需要考虑在当前层的作用
	# 这里的卷积层输入为 28*28*1 的原始 MNIST 图片像素。
	# 因为卷积层使用了全0填充，所以输出为 28*28*32
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
		
		# 使用边长为5，深度为32的过滤器，过滤器的移动步长为1，且使用全0填充
		conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
		relu1 =  tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
		
	# 实现第二层池化层的前向传播过程。
	# 这里选用最大池化层，池化层过滤器的边长为2，使用全0填充且移动步长为2.
	# 输入是上一层的输出。输入：28*28*32	输出：14*14*32
	with tf.name_scope('layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
		
	# 声明第三层卷积层的变量并实现前向传播过程
	# 这一层输入为 14*14*32		输出为 14*14*64
	with tf.variable_scope('layer3-conv2'):
		conv2_weights = tf.get_variable("weight",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
		
		# 使用边长为5，深度为64的过滤器，过滤器的移动步长为1，且使用全0填充
		conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
		
	# 实现第四层池化层的前向传播过程。
	# 这一层输入为 14*14*64		输出为 7*7*64
	with tf.name_scope('layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
	
	# 将第四层池化层的输出转化为第五层全连接层的输入格式。
	# 第四层输出为  7*7*64的矩阵，第五层全连接层需要的输入格式为向量，所以在这里需要把
	# 7*7*64 的矩阵拉成一个向量。
	# pool2.get_shape 函数可以得到第四层输出矩阵的维度而不需要手工计算。
	# 注意每一层神经网络的输入输出都为一个 batch 的矩阵，所以这里的维度也包含了一个 batch 中的数据个数
	pool_shape = pool2.get_shape().as_list()
	# 计算将矩阵拉成向量之后的长度，这个长度就是矩阵长宽以及深度的乘积。
	# pool_shape[0] 是 batch 中数据的个数 pool_shape[3]是深度
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	
	# 通过 tf.reshape 函数将第四层的输出变成一个 batch 的向量
	# reshaped.shape 可以查看 Tensor 的维度
	reshaped = tf.reshape(pool2,[pool_shape[0],nodes])
	# 声明第五层全连接层的变量并实现前向传播过程。输入是拉直之后的一组向量
	# 向量长度是 7*7*64(3136) 输出是 长度为512的向量。
	# 这里引入了 dropout 的方法。dropout 在训练时会随机将部分节点的输出改为0
	# dropout 可以避免过拟合，从而使得模型在测试数据上的效果更好
	# dropout 一般只在全连接层而不是卷积层或者池化层使用
	with tf.variable_scope('layer5-fc1'):
		fc1_weights = tf.get_variable("weight",[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
		# 只有全连接层的权重需要加入正则化
		if regularizer != None:
			tf.add_to_collection('losses',regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias",[FC_SIZE],initializer=tf.constant_initializer(0.1))
		
		fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
		if train:
			fc1 = tf.nn.dropout(fc1,0.5)
		
	# 声明第六层全连接层的变量并实现前向传播过程。
	# 这一层输入为一组长度为 512 的向量，输出为一组长度为 10 的向量。
	with tf.variable_scope('layer6-fc2'):
		fc2_weights = tf.get_variable("weight",[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None:
			tf.add_to_collection('losses',regularizer(fc2_weights))
		fc2_biases = tf.get_variable("bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc1,fc2_weights) + fc2_biases
	
	# 返回第六层的输出
	return logit
		