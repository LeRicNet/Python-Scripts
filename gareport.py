import pickle
import pandas as pd
import argparse

class GAReport:
	"""Breaks down dictionary data from GA to investigate population 
	parameters and accuracy.
	
	Usage:
		python3 GAReport.py \
			--pkl <path/to/pickle.p> \
			--out_fpath <path/to/csv.csv
	
	Returns:
		csv file containing all model parameters and accuracy for 
		each network of each generation from the GA.
	
	"""

	def __init__(self):
		self.FLAGS = self.fetch_args()
		self.data = None
		super().__init__()

	def fetch_args(self):
		parser = argparse.ArgumentParser()
		parser.add_argument(
			'--pkl', type=str, required=True,
			help='pickle file containing run dict'
		)
		parser.add_argument(
			'--out_fpath', type=str, required=True,
			help='path to save resulting CSV file'
		)
		args, _ = parser.parse_known_args()
		return args

	def fetch_data(self):
		ddict = pickle.load(
			open(
				self.FLAGS.pkl, 'rb'
			))
		new_dict = dict()
		for gen in ddict.keys():
			gen_data = ddict[gen]
			for i in range(len(gen_data)):
				idx = '{}::{}'.format(gen.split('_')[1], i)
				new_dict[idx] = gen_data[
					i].nn_params
				new_dict[idx]['acc'] = gen_data[i].accuracy

		df = pd.DataFrame(new_dict)
		df = df.transpose()
		df['idx'] = df.index.values
		# Splitting idx col for easier grouping downstream
		df['gen'], df['net'] = df['idx'].str.split('::', 1).str
		self.data = df
		return self

	def write_new_data(self):
		self.data.to_csv(self.FLAGS.out_fpath)

	def report(self):
		self.fetch_data()
		self.write_new_data()


if __name__ == '__main__':
	gar = GAReport()
	gar.report()
