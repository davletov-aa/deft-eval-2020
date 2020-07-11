import fire
from utils.data_preprocessing import (
	create_local_dataset_folder, create_multitask_dataset
)


def prepare_dataset(deft_corpus_repo: str = 'deft_corpus'):
	create_local_dataset_folder(
		f'{deft_corpus_repo}/task1_converter.py',
		f'{deft_corpus_repo}/data/deft_files/',
		f'{deft_corpus_repo}/local_data/',
		f'{deft_corpus_repo}/data/test_files/labeled/'
	)
	create_multitask_dataset(
		f'{deft_corpus_repo}/local_data/',
		'data'
	)


if __name__ == '__main__':
	fire.Fire(prepare_dataset)
