# coding:utf-8
# 训练神经网络解决二分类问题

import tensorflow as tf

# 通过 NumPy 生成工具包模拟数据集
from numpy.random import RandomState

# 定义训练数据 batch 的大小
batch_size = 8

# 定义神经网络的结构 w1=shape(2,3) w2=shape(3,1)
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 在 shape 的一个维度上使用 None 可以方便使用不同的 batch 大小。在训练时
# 需要把数据分成比较小的 batch ，但是在测试时，可以一次性使用全部的数据。
# 当数据集比较小时，这样比较方便测试。
# 但数据集比较大时，大量数据放入一个 batch 容易导致内存溢出
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

# 定义神经网络前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# 定义学习速率
learning_rate = 0.001
 
# 定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_ *tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)

# 定义规则来给样本的标签。在这里所有x1+x2<1的样例都被认为是正样本
# 而其他为负样本 
# 0表示负样本，1表示正样本
Y = [[int(x1+x2 < 1)]  for (x1,x2) in X]

# 创建一个会话来运行Tensorflow程序
with tf.Session() as sess:
	# 初始化变量
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print('Before training, the weigths are: ')
	print(sess.run(w1))
	print(sess.run(w2))
	
	# 设定训练的轮数
	STEPS = 5000
	for i in range(STEPS):
		# 每次选取 batch_size 大小个样本进行训练
		start = (i * batch_size) % dataset_size
		end = min(start+batch_size,dataset_size)
		
		# 通过选取的样本训练神经网络并更新参数
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i % 1000 == 0:
			# 每一千次迭代计算在所有数据上的交叉熵并输出
			total_cross_entropy = sess.run(cross_entropy,feed_dict={x:X,y_:Y})
			print('After %d training step(s), cross_entropy on all data is %g' % (i,total_cross_entropy))
		
	print('Before training, the weigths are: ')
	print(sess.run(w1))
	print(sess.run(w2))
	
			
	














































