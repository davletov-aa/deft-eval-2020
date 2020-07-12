# deft-eval-2020
To install dependencies:
```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```
To prepare dataset:

```bash
git clone https://github.com/adobe-research/deft_corpus.git
python prepare_dataset.py --deft_corpus_repo path_to_deft_corpus_repo
```

To construct hyperparams grid:
```bash
python grid_search.py create train_config.json hyperparams.json
```

To run grid search:
```bash
python grid_search.py search train_config.json hyperparams.json --device_id gpu_id
```