import fire
import json
import os

from itertools import product


hyperparams = [
	'learning_rate',
	'weight_decay',
	'dropout',
	'sent_type_clf_weight',
	'tags_sequence_clf_weight',
	'relations_sequence_clf_weight'
]
abbrs = [
	'lr', 'wd', 'drp', 'w1', 'w2', 'w3'
]


def search(config_path: str, grid_path: str, device_id: str = ''):
	config = json.load(open(config_path))
	grid = json.load(open(grid_path))
	device = config.pop('cuda_device')
	if device_id:
	   device = device_id

	output_dir = config.pop('output_dir')

	default_cmd = [
		f'CUDA_VISIBLE_DEVICES={device} python run_defteval.py'
	]
	for key, value in config.items():
		if isinstance(value, bool):
			if value:
				default_cmd.append(f'--{key}')
			continue
		if key == 'test_file':
			continue
		default_cmd.append(f'--{key} {value}')

	params = [
		[
			(key, value) for value in values
		] for key, values in grid.items()
	]

	for cur_params in product(*params):
		cur_params_dict = {}
		cmd = [x for x in default_cmd]

		for (param, value) in cur_params:
			cur_params_dict[param] = value
			cmd.append(f'--{param} {value}')

		if all([
			float(cur_params_dict[param]) == 0.0 for param in [
				'sent_type_clf_weight',
				'tags_sequence_clf_weight',
				'relations_sequence_clf_weight'
			]
		]):
			print('\nskipping point with all tasks weights set to 0 ...\n')
			continue

		model_dir = "-".join([f'{n}-{cur_params_dict[x]}' for x, n in zip(hyperparams, abbrs)])
		cmd.append(f'--output_dir {os.path.join(output_dir, model_dir)}')
		cmd = ' '.join(cmd)

		os.system(cmd)


def create(grid_path: str):

	grid = {}

	if os.path.exists(grid_path):
		raise ValueError(f'{grid_path} already exists')

	for hyperparam in hyperparams:
		cmd = input(f'enter separated by space values for search for |{hyperparam}|: ')
		if cmd == 'q':
			return
		hyper_values = cmd.split(' ')
		grid[hyperparam] = hyper_values

	json.dump(grid, open(grid_path, 'w'), indent=2)


def main(cmd: str, config_path: str, grid_path: str, device_id: str = ''):
	if cmd == 'search':
		search(config_path, grid_path, device_id)
	else:
		create(grid_path)


if __name__ == '__main__':
	fire.Fire(main)
