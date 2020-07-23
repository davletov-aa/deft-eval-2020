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


def search(
	itertation_id: int, config_path: str, grid_path: str,
	device_id: str = '', default_test_file: str = "true"
):
	config = json.load(open(config_path))
	grid = json.load(open(grid_path))
	device = config.pop('cuda_device')
	if device_id:
	   device = device_id

	output_dir = config.pop('output_dir')

	default_cmd = [
		f'python+run_defteval.py'
	]
	for key, value in config.items():
		if isinstance(value, bool):
			if value:
				default_cmd.append(f'--{key}')
			continue
		if key == 'test_file' and default_test_file.lower() == 'true':
			continue
		default_cmd.append(f'--{key}+{value}')

	params = [
		[
			(key, value) for value in values
		] for key, values in grid.items()
	]

	iteration = 0
	for cur_params in product(*params):
		cur_params_dict = {}
		cmd = [x for x in default_cmd]

		for (param, value) in cur_params:
			cur_params_dict[param] = value
			cmd.append(f'--{param}+{value}')

		if all([
			float(cur_params_dict[param]) == 0.0 for param in [
				'sent_type_clf_weight',
				'tags_sequence_clf_weight',
				'relations_sequence_clf_weight'
			]
		]):
			continue

		model_dir = "-".join([f'{n}-{cur_params_dict[x]}' for x, n in zip(hyperparams, abbrs)])

		# if os.path.exists(os.path.join(output_dir, model_dir)):
		# 	continue

		iteration += 1
		cmd.append(f'--output_dir+{os.path.join(output_dir, model_dir)};')
		cmd = '+'.join(cmd)

		if iteration == itertation_id:
			return cmd

	return 'sleep+10s;'

if __name__ == '__main__':
    fire.Fire(search)
