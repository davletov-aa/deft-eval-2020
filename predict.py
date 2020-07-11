import fire
from glob import glob
import os
import json


def predict(
	models_regexp: str,
	test_files_regexp: str,
	test_config_path: str,
	device_id: str
):
	models = glob(models_regexp)
	test_files = glob(test_files_regexp)
	print('found models:')
	print('\n'.join(models))
	print('found test files:')
	print('\n'.join(test_files))

	test_config = json.load(open(test_config_path))
	default_cmd = [
		f'CUDA_VISIBLE_DEVICES={device_id} python run_defteval.py'
	]
	for key, value in test_config.items():
		if isinstance(value, bool):
			if value:
				default_cmd.append(f'--{key}')
			continue
		default_cmd.append(f'--{key} {value}')

	cmd = input('type yes to proceed: ')
	if cmd.strip().lower() == 'yes':
		for model in models:
			for test_file in test_files:
				cmd = [x for x in default_cmd]
				cmd += [f'--output_dir {model}', f'--test_file {test_file}']
				cmd = ' '.join(cmd)
				# os.system(cmd)
				print(cmd)
	else:
		print('cancelled')


if __name__ == '__main__':
	fire.Fire(predict)
