import fire
import os
from utils.data_processing import *


def score_task_12_predictions(
    models_regex: str,
    local_data_dir: str,
    comment: str = '',
    scores_dir: str = 'scores'
):
    pass


def score_task_2_predictions(
    models_regex: str,
    local_data_dir: str,
    comment: str = '',
    scores_dir: str = 'scores'
):
    pass


def score_task_123_predictions(
    models_regex: str,
    local_data_dir: str,
    comment: str = '',
    scores_dir: str = 'scores'
):
    path_to_scorer_script = 'evaluation/semeval2020_06_evaluation_main.py'
    eval_config = 'evaluation/configs/eval_test.yaml'

    for part in ['test', 'dev']:
        best_task_1_predictions_regex = \
            'bert_stype_tags_and_relations_first_slen_256_eval_perep_4_linear/*/best_sent_type*{part}.tsv'
        best_task_2_predictions_regex = \
            'bert_stype_tags_and_relations_first_slen_256_eval_perep_4_linear/*/best_tags_sequence*{part}.tsv'
        best_task_3_predictions_regex = \
            'bert_stype_tags_and_relations_first_slen_256_eval_perep_4_linear/*/best_relations_sequence*{part}.tsv'

        score_task_1_predictions(
            'evaluation/semeval2020_06_evaluation_main.py',
            os.path.join(local_data_dir, f'task_1/{part}'),
            eval_config=eval_config,
            best_task_1_predictions_regex,
            temp_output='temp_output',
            clean_output=True,
            scores_dir=f'{scores_dir}/task_1/{part}-123-maxscore-{comment}',
            pool_type='max_score'
        )

        score_task_1_predictions(
            path_to_scorer_script=path_to_scorer_script,
            os.path.join(local_data_dir, f'task_1/{part}'),
            eval_config=eval_config,
            best_task_1_predictions_regex,
            temp_output='temp_output',
            clean_output=True,
            scores_dir=f'{scores_dir}/task_1/{part}-123-ellections-{comment}',
            pool_type='ellections'
        )

        score_task_2_predictions(
            path_to_scorer_script=path_to_scorer_script,
            os.path.join(local_data_dir, f'task_2/{part}'),
            path_to_eval_config=eval_config,
            predictions_regex=best_task_2_predictions_regex,
            temp_output='temp_output',
            clean_output=True,
            scores_dir=f'{scores_dir}/task_2/{part}-123-maxscore-{comment}',
            pool_type: str = 'max_score'
        )

        score_task_2_predictions(
            path_to_scorer_script=path_to_scorer_script,
            os.path.join(local_data_dir, f'task_2/{part}'),
            path_to_eval_config=eval_config,
            predictions_regex=best_task_2_predictions_regex,
            temp_output='temp_output',
            clean_output=True,
            scores_dir=f'{scores_dir}/task_2/{part}-123-ellections-{comment}',
            pool_type: str = 'ellections'
        )


def main(
    tasks: str,
    models_regex: str,
    local_data_dir: str,
    comment: str = '',
    scores_dir: str = 'scores'
):
    assert tasks in ['12', '2', '123']:
    if tasks == '123':
        score_task_123_predictions(
            models_regex=models_regex,
            local_data_dir=local_data_dir,
            comment=comment,
            scores_dir=scores_dir,
        )
    elif tasks == '12':
        score_task_12_predictions(
            models_regex=models_regex,
            local_data_dir=local_data_dir,
            comment=comment,
            scores_dir=scores_dir,
        )
    else:
        score_task_2_predictions(
            models_regex=models_regex,
            local_data_dir=local_data_dir,
            comment=comment,
            scores_dir=scores_dir,
        )


if __name__ == '__main__':
    fire.Fire(main)
