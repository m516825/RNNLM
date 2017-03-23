import tensorflow as tf
import numpy as np
import csv
import progressbar as pb
import sys
import data_utils
from data_utils import VocabularyProcessor
import os
import _pickle as cPickle
from tensorflow.contrib import learn
from Bi_Dir_LSTM import Bi_Dir_LSTM
import time
import copy
import answer
from sklearn.metrics.pairwise import cosine_similarity


tf.flags.DEFINE_integer("RNN_dim", 256, "Dimension of RNN layer")
tf.flags.DEFINE_integer("epoch", 15, "training epoch")
tf.flags.DEFINE_integer("batch_size", 100, "batch size for one iteration")
tf.flags.DEFINE_integer("embedding_size", 200, "the pre-train word embedding size")
tf.flags.DEFINE_integer("evaluate_every", 200, "evaluate_every")
tf.flags.DEFINE_integer("layer", 3, "LSTM layers")
tf.flags.DEFINE_integer("shift", 3, "label shift")
tf.flags.DEFINE_integer("ans_type", 1, "answer prob calculation function")
tf.flags.DEFINE_integer("predict_every", 200, "predict_every")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "save model every")
tf.flags.DEFINE_integer("num_sampled", 5000, "number of negative sampling")
tf.flags.DEFINE_integer("dev_size", 200, "dev size")
tf.flags.DEFINE_float("lr", 0.87, "learning rate")
tf.flags.DEFINE_string("embedding_data", "./hw1_data/glove.6B.200d.txt", "glove pre-train word embedding")
tf.flags.DEFINE_string("test_data", "./hw1_data/testing_data.csv", "testing data path")
tf.flags.DEFINE_string("train_dir", "./hw1_data/Holmes_Training_Data/", "training data directory")
tf.flags.DEFINE_string("checkpoint_file", "", "checkpoint_file to be load")
tf.flags.DEFINE_string("w2v_data", "./prepro/w2p_W.dat", "word to vector matrix for our vocabulary")
tf.flags.DEFINE_string("prepro_train", "./prepro/train.dat", "tokenized train data's path")
tf.flags.DEFINE_string("vocab", "./vocab", "vocab processor path")
tf.flags.DEFINE_string("output", "./pred.csv", "output file")
tf.flags.DEFINE_boolean("prepro", True, "preprocess training data")
tf.flags.DEFINE_boolean("eval", False, "Evaluate testing data")
tf.flags.DEFINE_boolean("ensemble", False, "ensemble latest 5 model, given smallest model number")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

class Data(object):
	def __init__(self, train_data, lengths):
		self.current = 0
		self.dev_data = train_data[:FLAGS.dev_size]
		self.train_data = train_data[FLAGS.dev_size:]
		self.dev_lengths = lengths[:FLAGS.dev_size] 
		self.lengths = lengths[FLAGS.dev_size:]
		self.length = len(self.train_data)
	def next_batch(self, size):

		if self.current == 0:
			shuffle_index = np.random.permutation(np.arange(self.length))
			self.train_data = self.train_data[shuffle_index]
			self.lengths = self.lengths[shuffle_index]

		if self.current + size < self.length:
			train_x, train_y = self.train_data[self.current:self.current+size,0:-1], self.train_data[self.current:self.current+size,FLAGS.shift:]
			seq_len = self.lengths[self.current:self.current+size]
			self.current += size
		else: 
			train_x, train_y = self.train_data[self.current:,0:-1], self.train_data[self.current:,FLAGS.shift:]
			seq_len = self.lengths[self.current:]
			self.current = 0

		return train_x, train_y, seq_len

class Language_Model(object):
	def __init__(self, vocab_processor, max_length, w2v_W, data):
		self.epoch = FLAGS.epoch
		self.batch_size = FLAGS.batch_size
		self.lr = FLAGS.lr
		self.RNN_dim = FLAGS.RNN_dim
		self.embedding_size = FLAGS.embedding_size
		self.vocab = vocab_processor._reverse_mapping
		self.vocab_size = len(self.vocab)
		self.max_length = max_length
		self.num_sampled = FLAGS.num_sampled
		self.layer = FLAGS.layer
		self.w2v_W = w2v_W
		self.data = data
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = config)
		self.test = data_utils.load_n_process_test(FLAGS.test_data, vocab_processor)
		self.gen_path()
		self.answer = answer.Answer()

	def gen_path(self):

		# Output directory for models and summaries
		timestamp = str(time.strftime('%b-%d-%Y-%H-%M-%S'))
		self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
		print ("Writing to {}\n".format(self.out_dir))

	    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
	def predict_ans(self, test):

		choices = test['format_choices']
		train_x = choices[:,:-1]
		train_y = choices[:,FLAGS.shift:]

		scores_1 = np.zeros(len(choices))

		seq_len = np.ones(1) * (test['lengths'] - 1)
		pre_ans = train_y[:,test['index']-1]
		outputs = self.sess.run(self.LSTM.concat_shift_outputs, 
			feed_dict={self.LSTM.seq_len:seq_len, self.LSTM.train_x:[train_x[0]], self.LSTM.output_keep_prob:1.0})
		prds = outputs[0,test['index']-1]
		prds_flat = np.reshape(prds, [-1, FLAGS.RNN_dim])

		softmax_fw = self.sess.run(self.LSTM.softmax_fw, feed_dict={self.LSTM.pred_x_list_s:prds_flat})[0]
		for c in range(len(choices)):
			scores_1[c] += np.log(softmax_fw[pre_ans[c]])

		ans_index_1 = np.argmax(scores_1)


		ans = ['a', 'b', 'c', 'd', 'e']

		train_x = choices[:,:-1]
		train_y = choices[:,FLAGS.shift:]
		weights = np.ones([self.max_length-FLAGS.shift, 5])
		seq_len = np.ones(len(choices)) * (self.max_length-1)

		v_loss = self.sess.run(self.LSTM.v_loss, 
			feed_dict={self.LSTM.seq_len:seq_len, 
					self.LSTM.train_x:train_x, 
					self.LSTM.train_y:train_y, 
					self.LSTM.output_keep_prob:1.0,
					self.LSTM.weights:weights
					})

		ans_index = np.argmin(v_loss)

		return ans[ans_index], ans[ans_index_1]

	def predict(self):
		ans_idx = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4}
		answers, index = self.answer.get_sampled_ans(100)

		pred_ans_1 = []
		pred_ans_2 = []
		for i in index:
			test = self.test[i]
			ans1, ans2 = self.predict_ans(test)
			pred_ans_1.append(ans1)
			pred_ans_2.append(ans2)
		
		acc1 = 0.
		acc2 = 0.

		for i in range(len(answers)):
			if answers[i] == pred_ans_1[i]:
				acc1 += 1.
			if answers[i] == pred_ans_2[i]:
				acc2 += 1.
		print ("Answer, accuracy: {}, {}".format(acc1/len(answers), acc2/len(answers)))

	def eval(self, data=None, lengths=None):
		if data == None:
			dev_data = self.data.dev_data[:,:-1]
			dev_ans = self.data.dev_data[:,FLAGS.shift:]
			seq_len = np.ones(len(self.data.dev_data)) * (self.max_length-1)
			lengths = self.data.dev_lengths
			# seq_len = self.data.dev_lengths - 1
		else:
			dev_data = data[:,:-1]
			dev_ans = data[:,FLAGS.shift:]
			# seq_len = lengths - 1
			seq_len = np.ones(len(self.data.dev_data)) * (self.max_length-1)

		outputs_s = self.sess.run(self.LSTM.concat_shift_outputs, 
			feed_dict={self.LSTM.seq_len:seq_len, self.LSTM.train_x:dev_data, self.LSTM.output_keep_prob:1.0})
		
		prd_s_out = None
		ans_out = None
		for idx in range(FLAGS.dev_size):
			prd_s, ans = outputs_s[idx], list(dev_ans[idx])
			if prd_s_out is None:
				prd_s_out = prd_s[:lengths[idx]-1]
				ans_out = ans[:lengths[idx]-1]
			else:
				prd_s_out = np.concatenate((prd_s_out, prd_s[:lengths[idx]-1]), axis=0)
				ans_out += ans[:lengths[idx]-1]

		equal_fw = self.sess.run(self.LSTM.equal_fw, 
			feed_dict={self.LSTM.pred_x_list_s:prd_s_out, self.LSTM.pred_y_list:ans_out})

		return np.mean(equal_fw)

	def build_model(self):
		
		with self.sess.as_default():
			self.LSTM = Bi_Dir_LSTM( 
									hidden_dim=self.RNN_dim, 
									vocab_size=self.vocab_size, 
									max_length=self.max_length,
									batch_size=self.batch_size,
									embedding_size=self.embedding_size, 
									word2vec_weight=self.w2v_W, 
									neg_sample_num=self.num_sampled,
									layer=self.layer,
									shift=FLAGS.shift)
			self.global_step = tf.Variable(0, name="global_step", trainable=False)

			# self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.LSTM.cost, global_step=self.global_step)

			self.optimizer = tf.train.AdagradOptimizer(self.lr)

			tvars = tf.trainable_variables()

			grads = []
			for grad in tf.gradients(self.LSTM.cost, tvars):
				if grad is not None:
					grads.append(tf.clip_by_norm(grad, 5))
				else:
					grads.append(grad)

			# self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.LSTM.cost, tvars), 5)

			self.updates = self.optimizer.apply_gradients(
    				zip(grads, tvars), global_step=self.global_step)

			self.saver = tf.train.Saver(tf.global_variables())

	def train(self):
		
		batch_num = self.data.length//self.batch_size if self.data.length%self.batch_size==0 else self.data.length//self.batch_size + 1

		with self.sess.as_default():
			if FLAGS.checkpoint_file == "":
				self.sess.run(tf.global_variables_initializer())
			else:
				self.saver.restore(sess, FLAGS.checkpoint_file)

			for ep in range(self.epoch):
				print ("epoch: {}".format(ep+1))
				cost = 0.
				
				for b in range(batch_num):
					train_x, train_y, seq_len = self.data.next_batch(self.batch_size)

					# index = [inx+bi*(self.max_length-1) for bi in np.arange(len(seq_len)) for inx in range(seq_len[bi]-1)]

					seq_len = np.ones(len(seq_len)) * (self.max_length-1)
					
					feed_dict={
								self.LSTM.train_x:train_x, 
								self.LSTM.train_y:train_y, 
								self.LSTM.seq_len:seq_len,
								self.LSTM.output_keep_prob:0.7
								}

					loss, _, step = self.sess.run([self.LSTM.cost, self.updates, self.global_step], feed_dict=feed_dict)
					
					current_step = tf.train.global_step(self.sess, self.global_step)

					cost += loss

					if current_step % FLAGS.evaluate_every == 0:
						print ("epoch {}, current_step {}, batch {}/{}, cost: {:.8}".format(ep+1, current_step, b+1, batch_num, cost/(b+1)))
					if current_step % FLAGS.predict_every == 0:
						accuarcy_fw = self.eval()
						accuarcy_t = self.eval(self.data.train_data[:200], self.data.lengths[:200])
						print ("Predict Next Accuracy:\n {}, {}".format(accuarcy_fw, accuarcy_t))
						self.predict()
					if current_step % FLAGS.checkpoint_every == 0:
						path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
						print ("\nSaved model checkpoint to {}\n".format(path))
					
				print (">>cost: {}".format(cost/batch_num))

def load_model(graph, sess, checkpoint_file):
	saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
	saver.restore(sess, checkpoint_file)

	train_x = graph.get_operation_by_name("train_x").outputs[0]
	train_y = graph.get_operation_by_name("train_y").outputs[0]
	seq_len = graph.get_operation_by_name("seq_len").outputs[0]
	pred_x_list_s = graph.get_operation_by_name("pred_x_list_s").outputs[0]
	pred_y_list = graph.get_operation_by_name("pred_y_list").outputs[0]
	output_keep_prob = graph.get_operation_by_name("output_keep_prob").outputs[0]
	concat_shift_outputs = graph.get_operation_by_name("LSTM/concat_shift_outputs").outputs[0]
	softmax_fw = graph.get_operation_by_name("Softmax/softmax_fw").outputs[0]
	v_loss = graph.get_operation_by_name("Pred/loss_pred").outputs[0]
	weights = graph.get_operation_by_name("weights").outputs[0]

	return {
			'train_x':train_x,
			'train_y':train_y,
			'seq_len':seq_len,
			'concat_shift_outputs':concat_shift_outputs,
			'softmax_fw':softmax_fw,
			'output_keep_prob':output_keep_prob,
			'pred_y_list':pred_y_list,
			'pred_x_list_s':pred_x_list_s,
			'weights':weights,
			'v_loss':v_loss
			}

def predict_ans(sess, model, test):

	ans = ['a', 'b', 'c', 'd', 'e']

	choices = test['format_choices']
	max_length = len(choices[0])

	train_x = choices[:,:-1]
	train_y = choices[:,FLAGS.shift:]

	scores_1 = np.zeros(len(choices))

	seq_len = np.ones(1) * (test['lengths'] - 1)
	pre_ans = train_y[:,test['index']-1]
	outputs = sess.run(model['concat_shift_outputs'], 
		feed_dict={model['seq_len']:seq_len, model['train_x']:[train_x[0]], model['output_keep_prob']:1.0})
	prds = outputs[0,test['index']-1]
	prds_flat = np.reshape(prds, [-1, FLAGS.RNN_dim])

	softmax_fw = sess.run(model['softmax_fw'], feed_dict={model['pred_x_list_s']:prds_flat})[0]
	for c in range(len(choices)):
		scores_1[c] += np.log(softmax_fw[pre_ans[c]])
	ans_index_1 = np.argmax(scores_1)

	train_x = choices[:,:-1]
	train_y = choices[:,FLAGS.shift:]
	weights = np.ones([max_length-FLAGS.shift, 5])
	seq_len = np.ones(len(choices)) * (max_length-1)

	v_loss = sess.run(model['v_loss'], 
		feed_dict={model['seq_len']:seq_len, 
				model['train_x']:train_x, 
				model['train_y']:train_y, 
				model['output_keep_prob']:1.0,
				model['weights']:weights
				})

	ans_index = np.argmin(v_loss)

	return (ans[ans_index], -v_loss), (ans[ans_index_1], np.exp(scores_1))

def dump_ans(answers):
	with open(FLAGS.output, 'w') as f:
		f.write("id,answer\n")
		for i, a in enumerate(answers):
			f.write(str(i+1)+','+str(a)+'\n')

def main(_):

	print ("\nParameters:")
	for attr, value in sorted(FLAGS.__flags.items()):
		print ("{}={}".format(attr, value))
	print ()

	if not os.path.exists('./prepro/'):
		os.makedirs('./prepro/')

	if not FLAGS.eval:
		if FLAGS.prepro:
			print ("Start preprocessing data...")
			vocab_processor, train_data, lengths = data_utils.pre_process_train(FLAGS.train_dir, FLAGS.test_data, FLAGS.prepro_train, FLAGS.vocab)
			print ("Vocabulary size(including test data): {}".format(len(vocab_processor._reverse_mapping)))

			print ("Start dumping word2vec matrix...")
			w2v_W = data_utils.build_w2v_matrix(vocab_processor, FLAGS.w2v_data, FLAGS.embedding_data, FLAGS.embedding_size)
			print ("Done dumping word2vec matrix")

		else:
			train_data, lengths = cPickle.load(open(FLAGS.prepro_train, 'rb'))
			vocab_processor = VocabularyProcessor.restore(FLAGS.vocab)
			w2v_W = cPickle.load(open(FLAGS.w2v_data, 'rb'))

		print ("Training data size: {}".format(train_data.shape))
		print ("Word2vec matrix size: {}".format(w2v_W.shape))

		np.random.seed(87)
		shuffle_index = np.random.permutation(np.arange(len(train_data)))
		train_data = train_data[shuffle_index]
		lengths = lengths[shuffle_index]

		data = Data(train_data, lengths)

		model = Language_Model(vocab_processor, len(train_data[0]), w2v_W, data)

		model.build_model()

		model.train()
	else:
		answers = []
		assert FLAGS.checkpoint_file != "", "you need to specify checkpoint_file"
		vocab_processor = VocabularyProcessor.restore(FLAGS.vocab)
		s_test = data_utils.load_n_process_test(FLAGS.test_data, vocab_processor)

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True

		if FLAGS.ensemble:
			print ("Ensemble five latest model...")
			answer_list = ['a', 'b', 'c', 'd', 'e']
			current_id = int(FLAGS.checkpoint_file.split('-')[-1])
			prefix = '-'.join(FLAGS.checkpoint_file.split('-')[:-1])
			checkpoint_files = [prefix+'-'+str(current_id+i*FLAGS.checkpoint_every) for i in range(5)]
			ensemble_scores = np.zeros((len(s_test), 5)) 

			for checkpoint_file in checkpoint_files:
				graph = tf.Graph()
				with graph.as_default(), tf.Session(config = config) as sess:
					model = load_model(graph, sess, checkpoint_file)
					print ("Predict answer using model, {}".format(checkpoint_file))
					pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(s_test)).start()
					for i, test in enumerate(s_test):
						ans1, ans2 = predict_ans(sess, model, test)
						if FLAGS.ans_type == 1:
							ans, scores = ans1[0], ans1[1]
						else:
							ans, scores = ans2[0], ans2[1]
						ensemble_scores[i] += scores
						pbar.update(i+1)
					pbar.finish()
				tf.reset_default_graph()

			for i in range(len(ensemble_scores)):
				max_idx = np.argmax(ensemble_scores[i])
				answers.append(answer_list[max_idx])

		else:
			graph = tf.Graph()
			with graph.as_default(), tf.Session(config = config) as sess:
				model = load_model(graph, sess, FLAGS.checkpoint_file)

				pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(s_test)).start()
				for i, test in enumerate(s_test):
					ans1, ans2 = predict_ans(sess, model, test)
					if FLAGS.ans_type == 1:
						ans = ans1[0]
					else:
						ans = ans2[0]
					answers.append(ans)
					pbar.update(i+1)
				pbar.finish()

		dump_ans(answers)


if __name__ == "__main__":
	tf.app.run()