# 用神经网络来识别 MINIST 数字

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集相关的常数
INPUT_NODE = 784	# 输入层的节点数
OUTPUT_NODE = 10	# 输出层的节点数 这里是0-9这10个输出数字

# 配置神经网络的参数
LAYER1_NODE = 500	# 隐藏层节点数
BATCH_SIZE = 100	# 一个训练batch中的训练数据个数 
					# 数字越大，越接近梯度下降；数字越小，越接近随机梯度下降
LEARNING_RATE_BASE = 0.8	# 基础的学习速率
LEARNING_RATE_DECAY = 0.99	# 学习率的衰减率
REGULARIZATION_RATE = 0.0001	# 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000		# 训练轮数
MOVING_AVERAGE_DECAY = 0.99		# 滑动平均衰减率

# 一个辅助函数。给定神经网络的输入和所有参数。计算神经网络的前向传播结果
# 在这里定义了一个使用 ReLU 激活函数的三层全连接神经网络。
# 通过加入隐藏层实现了多层网络结构，通过 ReLU 激活函数实现了去线性化。
# 在这个函数中也支持传入用于计算参数平均值的类
# 方便在测试时使用滑动平均模型
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
	# 当没有提供滑动平均类时，直接使用参数当前的取值
	if avg_class == None:
		# 计算隐藏层的前向传播结果，这里使用 ReLU 激活函数
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
		
		# 计算输出层的前向传播结果。
		# 在计算损失函数时会一并计算 softmax 函数，所以不需要加入激活函数。
		# 而且不加入 softmax 不会影响预测结果。
		# 预测时使用的是不同类别对应节点输出值的相对大小，有没有 softmax 层没有影响
		# 因此在计算前向传播时可以不加入最后的 softmax 层
		return tf.matmul(layer1,weights2) + biases2
	
	else:
		# 首先使用 avg_class.average 函数来计算得出变量的滑动平均值
		# 然后再计算相应的神经网络前向传播结果
		layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1))
		return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)
	
# 训练模型的过程
def train(mnist):
	x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
	y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
	
	# 生成隐藏层参数
	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
	
	# 生成输出层参数
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
	
	# 计算当前参数下神经网络前向传播结果 这里设定用于计算滑动平均的类为 None
	y = inference(x,None,weights1,biases1,weights2,biases2)
	
	# 定义存储训练轮数的变量 这个变量不需要计算滑动平均值
	# 所以这里指定这个变量为不可训练的变量 trainable = False
	# 使用 TensorFlow 训练神经网络时，一般会把代表训练轮数的变量指定为不可训练的参数
	global_step = tf.Variable(0,trainable=False)
	
	# 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	
	# 在所有代表神经网络参数的变量上使用滑动平均。其他变量不需要
	# tf.trainable_variables 返回的就是图上集合 GraphKeys.TRAINABLE_VARIABLES 中的元素
	#  这个集合就是所有没有指定 trainable=False 的参数
	variable_averages_op = variable_averages.apply(tf.trainable_variables())
	
	# 计算使用了滑动平均之后的前向传播结果
	# 滑动平均不会改变变量本身的取值，而是维护一个影子变量来记录其滑动平均值
	average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
	
	# 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
	# 当分类中只有一个正确答案时，可以用这个函数来加入交叉熵的计算、
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	# 计算在当前 batch 中所有样例的交叉熵平均值
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	
	# 计算 L2 正则化损失函数
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	# 计算模型的正则化损失。 只计算神经网络边上的正则化损失，而不使用偏置项
	regularization = regularizer(weights1) + regularizer(weights2)
	# 总损失等于交叉熵损失和正则化损失的和
	loss = cross_entropy_mean + regularization
	# 设置指数衰减的学习率
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
	
	# 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数
	# 这里的损失函数包括了交叉熵损失和 L2 正则化损失
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	
	# 在训练神经网络模型时 每过一遍数据既需要通过反向传播来更新神经网络中的参数
	# 又要更新每个参数的滑动平均值
	# 为了一次完成多个操作 TensorFlow 提供了tf.control_dependencies 和 tf.group 两种机制
	with tf.control_dependencies([train_step,variable_averages_op]):
		train_op = tf.no_op(name='train')
	
	# 检验使用了滑动平均模型的神经网络前向传播结果是否正确
	# tf.argmax(y_,1) 计算了每个样例的预测结果 其中 average_y 是一个 batch_size*10 的二维数组
	# 每一行表示一个样例的前向传播结果。tf.argmax 的第二个参数"1"表示选取最大值的操作在行进行
	# 得到的结果是一个长度为 batch 的一维数组 数组的值就表示了每一个样例对应的数字的识别结果
	# tf.equal 判断两个张量的每一维是否相等 相等返回 True 否则返回 False
	correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
	# 将布尔型的数值转化为实数型，然后计算平均值，即模型在这组数据上的正确率。
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	
	# 初始化会话并开始训练
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		# 准备验证数据
		validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
		# 准备测试数据
		test_feed = {x:mnist.test.images, y_:mnist.test.labels}
		
		# 迭代地训练神经网络
		for i in range(TRAINING_STEPS):
			# 每 1000 轮输出一次在验证数据集上的测试结果
			if i % 1000 == 0:
				# 计算滑动平均模型在验证数据上的结果。由于 MNIST 数据集比较小，
				# 所以可以一次处理所有的验证数据。如果神经网络模型比较复杂或者
				# 验证数据比较大时，太大的 batch 会导致计算时间过长甚至是内存溢出。
				validate_acc = sess.run(accuracy,feed_dict=validate_feed)
				print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
			# 产生这一轮使用的一个 batch 的训练数据，并运行训练过程
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op,feed_dict={x:xs, y_:ys})
			
		# 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
		test_acc = sess.run(accuracy,feed_dict=test_feed)
		print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))
		
		
# 主程序入口
def main(argv=None):
	# 声明 MNIST 数据集的类，这个在初始化时会自动下载数据。 "/" 在文件名的后面
	mnist = input_data.read_data_sets("data/",one_hot=True)
	train(mnist)
	
# TensorFlow 提供的一个主程序入口，tf.app.run 会调用上面定义的 main 函数
if __name__ == '__main__':
	tf.app.run()












