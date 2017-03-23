import tensorflow as tf
from tensorflow.contrib import rnn
import math

class Bi_Dir_LSTM(object):
	def __init__(
		self, hidden_dim, vocab_size, max_length, batch_size,
		embedding_size, word2vec_weight, neg_sample_num, layer, shift=3):

		self.train_x = tf.placeholder(tf.int32, [None, max_length-1], name='train_x')
		self.train_y = tf.placeholder(tf.int32, [None, max_length-shift], name='train_y')
		self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
		# self.index = tf.placeholder(tf.int32, [None], name='index')
		self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')
		self.weights = tf.placeholder(tf.float32, [max_length-shift, 5], name='weights')

		self.pred_x_list_s = tf.placeholder(tf.float32, [None, hidden_dim], name='pred_x_list_s')
		self.pred_y_list = tf.placeholder(tf.int64, [None], name='pred_y_list')

		with tf.name_scope('Embedding_Weight'):
			self.W = tf.Variable(word2vec_weight, name='W')
			self.sent_embeddings = tf.nn.embedding_lookup(self.W, self.train_x)
			# self.sent_embeddings = tf.nn.dropout(self.sent_embeddings, self.output_keep_prob)
			

		with tf.name_scope('LSTM'):
			# forward direction cell
			self.lstm_fw_cell = rnn.BasicLSTMCell(hidden_dim, forget_bias=0.0, state_is_tuple=True)
			# self.lstm_fw_cell = rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob=self.output_keep_prob)
			# backward direction cell
			self.lstm_bw_cell = rnn.BasicLSTMCell(hidden_dim, forget_bias=0.0, state_is_tuple=True)
			# self.lstm_bw_cell = rnn.DropoutWrapper(self.lstm_bw_cell, output_keep_prob=self.output_keep_prob)

			self.lstm_fw_cells = rnn.MultiRNNCell([self.lstm_fw_cell]*layer, state_is_tuple=True)

			self.lstm_bw_cells = rnn.MultiRNNCell([self.lstm_bw_cell]*layer, state_is_tuple=True)

			f_init_state = self.lstm_fw_cells.zero_state(tf.shape(self.train_x)[0], dtype=tf.float32)
			b_init_state = self.lstm_bw_cells.zero_state(tf.shape(self.train_x)[0], dtype=tf.float32)

			# (?, 51, dim)
			self.outputs, self.states = tf.nn.dynamic_rnn(self.lstm_fw_cells, 
														self.sent_embeddings,  
														sequence_length=self.seq_len, dtype=tf.float32, 
														initial_state=f_init_state, 
														scope='BiLSTM')

			self.concat_shift_outputs = tf.identity(self.outputs[:,shift-1:,:], name='concat_shift_outputs')
			self.all_outputs_fw = tf.reshape(self.concat_shift_outputs, [-1, hidden_dim])
			self.all_train_y = tf.reshape(self.train_y, [-1, 1])


		with tf.variable_scope('NCE_Weight'):

			self.w_nce_fw = tf.Variable(tf.truncated_normal([vocab_size, hidden_dim], stddev=6 / math.sqrt(hidden_dim*2)), name='w_nce_fw')
			self.b_nce_fw = tf.Variable(tf.zeros([vocab_size]), name='b_nce_fw')

		with tf.name_scope('Loss'):

			self.nce_loss_fw = tf.nn.nce_loss(weights=self.w_nce_fw, biases=self.b_nce_fw, inputs=self.all_outputs_fw,
						labels=self.all_train_y, num_sampled=neg_sample_num, num_classes=vocab_size, name='nce_loss_fw')
			
			# self.loss = tf.gather(self.nce_loss_fw, self.index)
			# self.loss = tf.contrib.losses.compute_weighted_loss(self.nce_loss_fw, self.loss_weights)
			self.cost = tf.reduce_mean(self.nce_loss_fw, name='cost') 

		with tf.name_scope('Softmax'):
			self.softmax_fw = tf.nn.softmax(tf.matmul(self.pred_x_list_s, tf.transpose(self.w_nce_fw)) + self.b_nce_fw, name='softmax_fw')

		with tf.name_scope('Accuarcy'):
			
			argmax_fw = tf.argmax(self.softmax_fw, axis=1)
			self.equal_fw = tf.equal(argmax_fw, self.pred_y_list, name='equal_fw')

		with tf.name_scope('Pred'):

			logits = tf.matmul(self.all_outputs_fw, tf.transpose(self.w_nce_fw)) + self.b_nce_fw
			logits = tf.reshape(logits, [-1, max_length-shift, vocab_size])
			logits = tf.unstack(tf.transpose(logits, [1, 0 ,2]))
			labels = tf.unstack(tf.transpose(self.train_y, [1, 0]))
			weights = tf.unstack(self.weights)

			self.v_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
					logits,
					labels,
					weights, 
					name='vloss')

			self.loss_pred = tf.add(self.v_loss, 0, name='loss_pred')

			self.v_cost = tf.reduce_mean(self.v_loss, name='v_cost')




