import fire
from utils.data_preprocessing import (
	create_local_dataset_folder, create_multitask_dataset
)


def prepare_dataset():
	create_local_dataset_folder(
		'deft_corpus/task1_converter.py',
		'deft_corpus/data/deft_files/',
		'deft_corpus/local_data/',
		'deft_corpus/data/test_files/labeled/'
	)
	create_multitask_dataset(
		'deft_corpus/local_data/',
		'data'
	)


if __name__ == '__main__':
	fire.Fire(prepare_dataset)
