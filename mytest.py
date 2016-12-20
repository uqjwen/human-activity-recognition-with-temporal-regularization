import sys
from keras.layers import Dense
import tensorflow as tf 
import numpy as np 
from keras.utils.np_utils import to_categorical

datapath1 = './data/UCI HAR Dataset/train/'
datapath2 = './data/UCI HAR Dataset/test/'

class DataLoader():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.xtrain = np.genfromtxt(datapath1+'X_train.txt')
		ytrain = np.genfromtxt(datapath1+'y_train.txt').astype(np.int32)
		self.xtest = np.genfromtxt(datapath2+'X_test.txt')
		ytest = np.genfromtxt(datapath2+'y_test.txt').astype(np.int32)

		# print np.unique(ytrain-1)
		self.nb_classes = len(np.unique(ytrain))
		self.ytrain = to_categorical(ytrain-1, self.nb_classes)
		self.ytest = to_categorical(ytest-1, self.nb_classes)

		self.train_size, self.data_dims = self.xtrain.shape

	def reset_pointer(self):
		self.pointer = 0
	def next_batch(self):
		start = self.pointer*self.batch_size
		end = (self.pointer+1)*self.batch_size

		self.pointer+=1
		return self.xtrain[start:end], self.ytrain[start:end]




class Model():
	def __init__(self, batch_size, data_dims, nb_classes, hidden_size):
		self.batch_size = batch_size
		self.data_dims = data_dims
		self.nb_classes = nb_classes
		self.hidden_size = hidden_size

		self.x = tf.placeholder(tf.float32, shape = [self.batch_size, self.data_dims])
		self.y = tf.placeholder(tf.float32, shape = [self.batch_size, self.nb_classes])
		self.hidden_layer = Dense(output_dim = self.hidden_size, input_dim = self.data_dims, activation = 'tanh')(self.x)
		self.logits = Dense(output_dim = self.nb_classes, input_dim = self.hidden_size)(self.hidden_layer)
		# self.probs = Activation('softmax')(self.logits)
		self.probs = tf.nn.softmax(self.logits)

		mat = np.zeros((self.batch_size, self.batch_size))
		neighbors = 4
		for k in  range(-neighbors/2,neighbors/2+1):
			if k == 0:
				continue
			mat += np.eye(self.batch_size, k=k)
		
		matrix = tf.constant(mat/neighbors, tf.float32)

		self.loss = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y)

		self.neighbors = tf.mul(self.y, tf.matmul(matrix, self.logits))

		self.reg = tf.reduce_sum(tf.square(tf.mul(self.y, self.logits)-self.neighbors), 1)

		lam = 0.1

		self.cost = self.loss + lam*self.reg

		self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

		self.pred_logits = self.logits - lam*tf.square(self.logits - tf.matmul(matrix, self.logits))
		self.correct_prediction = tf.equal(tf.argmax(self.pred_logits, 1), tf.argmax(self.y,1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		# self.prediction = tf.argmax(self.pred_logits, dimension = 1)
		# print self.prediction.get_shape()
		# self.saver = tf.train.Saver(tf.all_variables())
	# def inspection(self):

def validation(sess, model, data_loader, batch_size = 64):
	xtest = data_loader.xtest
	ytest = data_loader.ytest
	test_size = len(xtest)
	total_batch = int(test_size/batch_size)
	accuracy = []
	for b in range(total_batch):
		begin = b*batch_size
		end = (b+1)*batch_size
		batch_x = xtest[begin:end]
		batch_y = ytest[begin:end]
		acc = sess.run(model.accuracy, feed_dict={model.x:batch_x, model.y:batch_y})
		accuracy.append(acc)
	return np.mean(accuracy)


if __name__ == '__main__':

	batch_size = 64
	epochs = 100
	data_loader = DataLoader(batch_size = batch_size)
	model = Model(batch_size = batch_size,
				data_dims = data_loader.data_dims,
				nb_classes = data_loader.nb_classes,
				hidden_size = 128)
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
		saver = tf.train.Saver(tf.all_variables())
		checkpoint_dir = './checkpoint/'
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print (' [*] Loaded parameters success!!!')
		else:
			print (' [!] Loaded parameters failed...')

		for e in range(epochs):
			total_batch = int(data_loader.train_size/batch_size)
			data_loader.reset_pointer()
			for b in range(total_batch):
				x,y = data_loader.next_batch()
				train_acc,_ = sess.run([model.accuracy, model.train_op], feed_dict={model.x:x, model.y:y})
				# val_acc = sess.run(model.accuracy, feed_dict={model.x: data_loader.xtest,
				# 											model.y: data_loader.ytest})
				sys.stdout.write('\r {}/{} epoch, {}/{} batch. train_acc:{}'.\
								format(e,epochs, b, total_batch, train_acc))
				sys.stdout.flush()
				# print ('\n'+str(train_acc))
				if (e*total_batch+b)%100==0 or\
					(e==epochs-1 and b == total_batch-1):
					saver.save(sess, checkpoint_dir+'model.ckpt', global_step = e*total_batch+b)
			val_acc = validation(sess, model, data_loader, batch_size)
			print (' val_acc:{}'.format(val_acc))




	# data_loader = DataLoader(batch_size = 64)