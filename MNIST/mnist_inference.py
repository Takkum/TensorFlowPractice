# coding:utf-8
import tensorflow as tf


# 定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 通过 tf.get_variable 函数来获取变量。在训练网络时创建这些变量 在测试时会通过保存的模型
# 加载这些变量。在加载变量时将滑动平均变量重命名，所以可以直接通过名字在训练时使用变量自身。
# 在测试时使用变量的滑动平均值。
# 这个函数中将正则化损失加入损失集合
def get_weight_variable(shape,regularizer):
	weights = tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
	
	# 当给出正则化生成函数时，将当前变量的正则化损失函数加入到 losses 的集合
	# 在这里使用 add_to_collection 函数将一个张量加入一个集合，集合名称为 losses
	if regularizer != None:
		tf.add_to_collection('losses',regularizer(weights))
	return weights

# 定义神经网络的前向传播过程
def inference(input_tensor,regularizer):
	with tf.variable_scope('layer1'):
		weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
		biases = tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
		
	with tf.variable_scope('layer2'):
		weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
		biases = tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1,weights)+biases
		
	return layer2