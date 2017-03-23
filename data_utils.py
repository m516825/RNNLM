import numpy as np
from tensorflow.contrib import learn
import collections
import os
import re
import csv
import sys
import _pickle as cPickle
from tensorflow.python.platform import gfile
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
try:
	import cPickle as pickle
except ImportError:
	import pickle

MAX_LENGTH = 40
MIN_LENGTH = 10
MIN_VOCAB_COUNT = 5
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
	if tag.startswith('J'):
		return wordnet.ADJ
	elif tag.startswith('V'):
		return wordnet.VERB
	elif tag.startswith('N'):
		return wordnet.NOUN
	elif tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN

class VocabularyProcessor(object):
	def __init__(self, max_document_length, vocabulary, unknown_limit=0):
		self.max_document_length = max_document_length
		self._reverse_mapping = ['<UNK>'] + vocabulary
		self._mapping = {'<UNK>': 0}
		self.make_mapping()
		self.unknown_limit = unknown_limit

	def make_mapping(self):
		for i, vocab in enumerate(self._reverse_mapping):
			self._mapping[vocab] = i

	def transform(self, raw_documents):
		data = []
		lengths = []
		for tokens in raw_documents:
			word_ids = np.ones(self.max_document_length, np.int32) * self._mapping['<END>']
			length = 0
			unknown = 0
			for idx, token in enumerate(tokens.split()):
				if idx >= self.max_document_length:
					break
				word_ids[idx] = self._mapping.get(token, 0)
				length = idx
				if word_ids[idx] == 0:
					unknown += 1
			length = length+1
			if unknown <= self.unknown_limit:
				data.append(word_ids)
				lengths.append(length)

		data = np.array(data)
		lengths = np.array(lengths)

		return data, lengths
			# yield word_ids
	def save(self, filename):
		with gfile.Open(filename, 'wb') as f:
			f.write(pickle.dumps(self))
	@classmethod
	def restore(cls, filename):
		with gfile.Open(filename, 'rb') as f:
			return pickle.loads(f.read())

def clean_str(string):

	string = re.sub(r"[^A-Za-z0-9().,!?\':]", " ", string)
	string = re.sub(r"\(.*?\)", " ", string)
	string = re.sub(r"\'m", " am", string)
	string = re.sub(r'([A-Z][A-Za-z]*)([.])( )', r'\1\3', string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"Can\'t", "can \'t", string)
	string = re.sub(r"can\'t", "can \'t", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " ( ", string)
	string = re.sub(r"\)", " ) ", string)
	string = re.sub(r"\?", " ? ", string)
	string = re.sub(r"\.", " . ", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"(.{3,})\'", r"\1", string)
	string = re.sub(r"\'(.{3,})", r"\1", string)


	return string.strip().lower()+' '

def add_test_vocab(test_data, vocab, must_vocab):
	total_test = []
	with open(test_data, 'r', encoding='utf-8') as f:
		for i, row in enumerate(csv.reader(f)):
			if i == 0:
				continue
			else:
				q_string = re.sub("_____", "",row[1])
				q_string = clean_str(q_string)
				pos_s = pos_tag(q_string.split())
				split_lemma_s = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_s]
				q_string = ' '.join(split_lemma_s)

				pos_s = pos_tag(row[2:])
				split_lemma_s = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_s]
				q_string += ' '+' '.join(split_lemma_s)
				for w in q_string.split():
					vocab[w] += 1
					must_vocab[w] = 1				
				
	return vocab, must_vocab

def is_end_mark(char):
	if char in ['.', '"','!', '?', ':']:
		return True
	else:
		return False

def pre_process_train(train_dir, test_data, prepro_path, vocab_path):

	total_sen = []
	vocab = collections.defaultdict(int)
	must_vocab = dict()
	for dirPath, dirNames, fileNames in os.walk(train_dir):
		for f in fileNames:
			filePath = os.path.join(train_dir, f)
			context = ''
			paragraph = ''
			start = False
			with open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
				for line in f.readlines():
					if start:
						if line == '\n':
							if len(paragraph) >=2 and is_end_mark(paragraph[-2]):
								context += paragraph
							paragraph = ''
						paragraph += clean_str(line.strip())
					if not start and line.find("*END*") >= 0:
						start = True
			sentences = re.split(r"[.?!;:]", context)
			# split long sentence
			split_sen = []
			for s in sentences:
				s = s.strip()
				if s == '':
					continue
				split_s = s.split()
				if len(split_s) <= MIN_LENGTH:
					continue
				pos_s = pos_tag(split_s)
				split_lemma_s = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_s]
				lemma_s = ' '.join(split_lemma_s)
				for w in split_lemma_s:
					vocab[w] += 1 
				vocab['<START>'] += 1
				vocab['<END>'] += 1
				if len(split_lemma_s) >= MAX_LENGTH-2:
					s_sen = lemma_s.split(',')
					split_sen += ['<START> '+s_+' <END>' for s_ in s_sen if len(s_.split()) > MIN_LENGTH]
				else:
					split_sen += ['<START> '+lemma_s+' <END>']
			
			total_sen += split_sen

	vocab, must_vocab = add_test_vocab(test_data, vocab, must_vocab)

	vocabulary = []
	for k, v in sorted(vocab.items(), key=lambda x:x[1], reverse=True):
		if v >= MIN_VOCAB_COUNT or must_vocab.get(k, 0) > 0:
			vocabulary.append(k)
	vocab_processor = VocabularyProcessor(max_document_length=MAX_LENGTH, vocabulary=vocabulary, unknown_limit=5)
	train_data, lengths = vocab_processor.transform(total_sen)

	np.random.seed(100)
	shuffle_index = np.random.permutation(np.arange(len(train_data)))
	train_data = train_data[shuffle_index]
	lengths = lengths[shuffle_index]
	print (vocab_processor._reverse_mapping[0:20])
	del vocab, total_sen

	vocab_processor.save(vocab_path)
	print (train_data.shape)
	# use  protocol=4 to dump large data
	cPickle.dump((train_data, lengths), open(prepro_path, 'wb'),  protocol=4)

	return vocab_processor, train_data, lengths

def get_unknown_word_vec(dim_size):
	return np.random.uniform(-0.25, 0.25, dim_size) 

def build_w2v_matrix(vocab_processor, w_path, embedding_path, dim_size):
	w2v_dict = {}
	f = open(embedding_path, 'r')
	for line in f.readlines():
		word, vec = line.strip().split(' ', 1)
		w2v_dict[word] = np.loadtxt([vec], dtype='float32')

	vocab_list = vocab_processor._reverse_mapping
	w2v_W = np.zeros(shape=(len(vocab_list), dim_size), dtype='float32')

	for i, vocab in enumerate(vocab_list):
		# unknown vocab
		if i == 0:
			continue
		else:
			if vocab in w2v_dict:
				w2v_W[i] = w2v_dict[vocab]
			else:
				w2v_W[i] = get_unknown_word_vec(dim_size)
				# print ("unkown word: {}".format(vocab))

	cPickle.dump(w2v_W, open(w_path, 'wb'),  protocol=4)

	return w2v_W

def load_n_process_test(test_data, vocab_processor):
	test = []
	with open(test_data, 'r', encoding='utf-8') as f:
		for i, row in enumerate(csv.reader(f)):
			test_struct = {'raw_question':"", 'q_question':"", 'choices':[], 'index':-1, 'format_choices':[]}
			if i == 0:
				continue
			else:
				test_struct['raw_question'] = row[1]
				pos_s = pos_tag(row[2:])
				split_lemma_s = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_s]
				test_struct['choices'] = split_lemma_s # without white space
				c_string = clean_str(row[1].split("_____")[0])+" _____ "+clean_str(row[1].split("_____")[1])
				c_string = c_string.replace(",", "")
				c_string = re.split(r"[.?!;:]", c_string)
				q_string = ""
				for string in c_string:
					s_str = string.split()
					if "_____" in s_str:
						q_string = string
						test_struct['index'] = s_str.index("_____") + 1 # adding <START>
						break
				pos_s = pos_tag(q_string.split())
				split_lemma_s = [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_s]
				q_string = ' '.join(split_lemma_s)
				test_struct['q_question'] = "<START> "+q_string+" <END>"
				format_choices = []
				for c in test_struct['choices']:
					# replace_c = re.sub("_____", c, test_struct['q_question'])
					replace_c = test_struct['q_question'].replace("_____", c)
					f_c, f_c_len = vocab_processor.transform([replace_c])
					format_choices.append(f_c[0])
				format_choices = np.array(format_choices)
				test_struct['format_choices'] = format_choices
				test_struct['lengths'] = f_c_len[0]

				test.append(test_struct)

	return test
