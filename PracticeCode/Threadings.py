# tf.Coordinator
import tensorflow as tf
import numpy as np
import threading 
import time

# three main functions: should_stop, request_stop, join


# If one thread call request_stop function, it will set should_stop true for all the threads 
# then all the threads can stop at next iteration certainly. 
def MyLoop(coord,worker_id):
	while not coord.should_stop():
		if np.random.rand() < 0.1:
			print('Stoping from id: %d' % worker_id)
			coord.request_stop()
		else:
			print('Working on id: %d' % worker_id)
		
		# stop 4 seconds
		time.sleep(4)

coord = tf.train.Coordinator()
threads = [threading.Thread(target=MyLoop,args=(coord,i)) for i in range(5)]

# start all the threads
for t in threads: 
	t.start()

# wait for all threads to quit
coord.join(threads)

		
