# ---------------------------
# Ex1:
# import tensorflow as tf
# c = tf.constant('Hello, distributed Tensorflow!')
# server = tf.train.Server.create_local_server()
# sess = tf.Session(server.target)
# print(sess.run(c))
# ---------------------------


# ---------------------------
# Ex2:
# import tensorflow as tf
# cluster = tf.train.ClusterSpec({"local":["localhost:2222","localhost:2223"]})
# server = tf.train.Server(cluster,job_name="local",task_index=0)
# sess = tf.Session(server.target,config=tf.ConfigProto(log_device_placement=True))
# print(sess.run(c))
# server.join()
# ---------------------------



# Between-graph replication 
# parameter server: store,get and update variables
# worker server: run back propagation to get gradients
# ---------------------------
# tf.train.ClusterSpec({
# 		"worker":[
#			"tf-worker0:2222",
#			"tf-worker1:2222",
#			"tf-worker2:2222",
#		],
#		"ps":[
#			"tf-ps0:2222",
#			"tf-ps1:2222"	
#		]})
#
# Note that the below tf-worker(i) and tf-ps(i) are the name of server address.
# ---------------------------

# ---------------------------
# Asynchronous Mode and Synchronous Mode
# Asynchronous Mode Code: one parameter server , two worker server.


# import time 
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data

# import mnist_inference
# BATCH_SIZE = 100
# LEARNING_RATE_BASE = 0.01
# LEARNING_RATE_DECAY = 0.99
# REGULARIZATION = 0.0001
# TRAINING_STEPS = 10000
# MODEL_SAVE_PATH = 'Tosave/model/'
# DATA_PATH = 'data/'
# 
# 通过 FALGS 指定运行的参数
# FLAGS = tf.app.flags.FLAGS  
# tf.app.flags.DEFINE_string('job_name','worker',' "ps" or "worker" ')
# 
# 指定集群中的参数服务器地址
# tf.app.flags.DEFINE_string(
#			'ps_hosts','tf-ps0:2222,tf-ps1:1111',
#			'Comma-separated list of hostname:port for the parameter server jobs.'
#			'e.g. "tf-ps0:2222,tf-ps1:1111" ')
# 
# 指定集群中的计算服务器地址
# tf.app.flags.DEFINE_string(
# 			'worker_hosts','tf-worker0:2222,tf-worker1:1111',
#			'Comma-separated list of hostname:port for the worker jobs.'
#			'e.g. "tf-worker0:2222,tf-worker1:1111" ')
# 
# 指定当前程序的ID
# tf.app.flags.DEFINE_integer('task_id',0,'Task ID of the worker/replica running the training.')
#
# 定义 Tensorflow 计算图
# def build_model(x,y,is_chief):
# 		......
# 		......
# 	return global_step,loss,train_op
#
# 训练分布式深度学习模型的主程序
# def main(argv=None):
# 	解析 flags 并通过 tf.train.ClusterSpec 配置 Tensorflow 集群
# 	ps_hosts = FLAGS.ps_hosts.split(',')
#	worker_hosts = FLAGS.worker_hosts.split(',')
#	cluster = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})
#	通过 ClusterSpec 以及当前任务创建 Server
#	server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_id)
# 	
# 	参数服务器只需要管理 Tensorflow 中的变量 不需要执行训练过程
# 	if FLAGS.job_name == 'ps':
#		server.join()
# 
#	定义计算服务器需要的操作。
#	在所有的计算服务器中，有一个是主计算服务器，它除了计算反向传播的结果还要输出日志和保存模型
#	is_chief = (FLAGS.task_id)
# 	mnist = input_data.read_data_sets(DATA_PATH,one_hot=True)
#	通过 tf.train.replica_device_setter 函数来指定执行每一个运算的设备
#	tf.train.replica_device_setter 函数会自动将所有的参数分配到参数服务器上，
#	把计算分配到当前的计算服务器上。
# 	with tf.device(tf.train.replica_device_setter(worker_device='/job:worker/task:%d' % FLAGS.task_id,cluster=cluster)):
#	x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name='x-input')
#	y = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
#	
#	定义训练模型需要的步骤
#	global_step,loss,train_op = build_model(x,y_)
#	
#	定义用于保存模型的 Saver
# 	saver = tf.train.Saver()
#
#	定义日志输出
#	summary_op = tf.summary.merge_all()
#	init_op = tf.global_variables_initializer()
#	通过 tf.train.Supervisor 能统一管理队列操作、模型保存、日志输出等
#	sv = tf.train.Supervisor(
#		is_chief=is_chief,
#		logdir = MODEL_SAVE_PATH,
#		init_op = init_op,
#		summary_op = summary_op,
#		saver = saver,
#		global_step = global_step,
#		save_model_secs = 100,
#		saver_summaries_secs = 100)
#		
#	sess_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
#	通过 tf.train.Supervisor 生成会话
#	sess = sv.prepare_or_wait_for_session(server.target,config=sess_config)
#
#	step = 0
#	start_time = time.time()
#	执行迭代过程
#	while not sv.should_stop():
#		xs,ys = mnist.train.next_batch(BATCH_SIZE)
#		_,loss_value,global_step_value = sess.run([train_op,loss,global_step],\
#		feed_dict={x:xs,y_:ys})
#		if global_step_value >= TRAINING_STEPS:
#			break
#
#		每隔一段时间输出训练信息
#		if step > 0 and step % 100 == 0:
#			duration = time.time() - start_time
#			不同的计算服务器都会更新全局的训练轮数，所以这里使用 global_step_value 可以直接
#			得到在训练中使用过的 batch 的总数
#			sec_per_batch = duration/global_step_value
#			format_str = ("After %d training steps (%d global steps),loss on training batch"
#					"is %g.(%.3f sec/batch)")
#			print(format_str % (step,global_step_value,loss_value,sec_per_batch))
#		step += 1
#	sv.stop()
# if __name__ == "__main__":
#	tf.app.run()
# ---------------------------

# ---------------------------
# file_name:dist_tf_mnist_async.py
# To start one ps, two worker
# First, you should run the following codes on the ps: python dist_tf_mnist_async.py --job_name='ps' --task_id=0 --ps_hosts='tf-ps0:2222' --worker_hosts='tf-worker0:2222,tf-worker1:2222' 
#
# Then, you run the codes on the first worker: python dist_tf_mnist_async.py --job_name='worker' --task_id=0 --ps_hosts='tf-ps0:2222' --worker_hosts='tf-worker0:2222,tf-worker1:2222'

# Last, you run the codes on the second worker: python dist_tf_mnist_async.py --job_name='worker' --task_id=1 --ps_hosts='tf-ps0:2222' --worker_hosts='tf-worker0:2222,tf-worker1:2222'

# 以上我都看不懂，就先照着书敲下来吧！
 


