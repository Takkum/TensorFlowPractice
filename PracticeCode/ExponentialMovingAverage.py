# 滑动平均模型

# shadow_variable = decay*shadow_variable + (1-decay)*variable
# decay = min{decay,(1+num_updates)/(10+num_updates)}

# 能使模型在测试数据上更健壮

import tensorflow as tf

# 定义一个变量用于计算滑动平均，这个变量的初始值为0。
# 所有需要计算滑动平均的变量必须是实数型
v1 = tf.Variable(0,dtype=tf.float32)

# 这里 step 变量模拟神经网络中的迭代轮数 用于动态控制衰减率
step = tf.Variable(0,trainable=False)

# 定义一个滑动平均的类 初始时给定衰减率0.99 和控制衰减率的变量 step
ema = tf.train.ExponentialMovingAverage(0.99,step)

# 定义一个更新变量滑动平均的操作。这里需要给定一个列表 每次执行这个操作时
# 这个列表中的变量都会被更新
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
	# 初始化所有变量
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	
	# 通过 ema.average(v1) 获取滑动平均值后变量的取值
	# 在初始化后  v1 的值和 v1 的滑动平均都为0
	print(sess.run([v1,ema.average(v1)]))
	
	# 更新 v1 的值为5
	sess.run(tf.assign(v1,5))
	# 更新 v1 的滑动平均值。 衰减率为 min{0.99,(1+step)/(10+step)=0.1}=0.1	
	# 所以 v1 的滑动平均会被更新为 0.1*0+0.9*5=4.5
	sess.run(maintain_average_op)
	print(sess.run([v1,ema.average(v1)]))
	
	# 更新 step 为 10000
	sess.run(tf.assign(step,10000))
	# 更新 v1 为10
	sess.run(tf.assign(v1,10))
	# 更新 v1 的滑动平均值。  衰减率为 min{0.99,(1+step)/(10+step)=0.999}=0.99
	# 所以 v1 的滑动平均会被更新为 0.99*4.5+0.01*10=4.555
	sess.run(maintain_average_op)
	print(sess.run([v1,ema.average(v1)]))
	
	# 再次更新滑动平均值，得到的心得滑动平均值为 0.99*4.555+0.01*10=4.60945
	sess.run(maintain_average_op)
	print(sess.run([v1,ema.average(v1)]))
	















