import fire
from utils.data_processing import (
	create_local_dataset_folder, create_multitask_dataset
)


def prepare_dataset(deft_corpus_repo: str = 'deft_corpus'):
	print('copying files ...')
	create_local_dataset_folder(
		f'{deft_corpus_repo}/task1_converter.py',
		f'{deft_corpus_repo}/data/deft_files/',
		f'{deft_corpus_repo}/local_data/',
		f'{deft_corpus_repo}/data/test_files/labeled/'
	)
	print('transforming data to multi-task format ...')
	create_multitask_dataset(
		f'{deft_corpus_repo}/local_data/',
		'data'
	)
	print('the dataset has been written to data folder')


if __name__ == '__main__':
	fire.Fire(prepare_dataset)
