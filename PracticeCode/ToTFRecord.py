# show how to translate MNIST data into TFRecord
# if the data is too large, you can write to some tfrecord files.

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 生成整数型的属性
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	
# 生成字符串型的属性
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
mnist = input_data.read_data_sets('data/',dtype=tf.uint8,one_hot=True)

# 55000 * 784
images = mnist.train.images

# 正确答案可以作为一个属性保存在 TFRecord 里
labels = mnist.train.labels

# 训练数据的图像分辨率作为 Example 中的一个属性
pixels = images.shape[1]	# 784
num_examples = mnist.train.num_examples	# 55000

# 输出 TFRecord 文件的地址
filename = 'output.tfrecords'

writer = tf.python_io.TFRecordWriter(filename)

for index in range(num_examples):
	# 将图像矩阵转化为一个字符串
	image_raw = images[index].tostring()
	# 将一个样例转化为 Example Protocol Buffer 并将所有的信息写入这个数据结构
	example = tf.train.Example(features=tf.train.Features(feature={
											'pixels':_int64_feature(pixels),
											'labels':_int64_feature(np.argmax(labels[index])),
											'image_raw':_bytes_feature(image_raw)}))
	
	# 将一个 Example 写入 TFRecord 文件
	writer.write(example.SerializeToString())

writer.close()




