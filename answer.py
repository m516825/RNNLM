import numpy as np
import csv

class Answer(object):
	def __init__(self):
		self.load_ans()

	def load_ans(self):
		self.ans = []
		f = open('./test_answer.csv', 'r')
		for i, row in enumerate(csv.reader(f)):
			if i == 0:
				continue
			else:
				self.ans.append(row[1])

	def get_sampled_ans(self, count):

		index = np.random.choice(len(self.ans), count, replace=False)

		return [self.ans[i] for i in index], index