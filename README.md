# deft-eval-2020
To install dependencies:
```bash
conda create -n env_name -c conda-forge -c anaconda -c pytorch python=3.7 requests numpy=1.17.2 pandas=1.0.3 fire=0.2.1 pytorch=1.4.0 future==0.18.2 tensorboardx==2.1
conda activate env_name
pip install fire tqdm python-Levenshtein
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