def check_floats(dir_path):
	file_list = os.listdir(dir_path)
	file_list = [os.path.join(dir_path, fl) for fl in file_list]
	pbar = tqdm.tqdm(file_list)
	n_procs = 10 # could also be os.cpu_count(), but I wanted to
	# avoid overloading memory
	def _worker_fn(fl):
		with open(fl, 'r') as f:
			s = f.read()
			s = s.split(',')
			s = [float(i) for i in s]
			if not all(isinstance(i, float) for i in s):
				print(fl + ' failed float test')
		pbar.update(1)
		del s
	with Pool(n_procs) as pool:
		pool.map(_worker_fn, file_list)
	pbar.close()
