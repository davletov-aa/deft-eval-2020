import fire
import os
from utils.data_processing import *
from glob import glob


def score_task_12(
    models_regex: str,
    local_data_dir: str,
    comment: str = '',
    scores_dir: str = 'scores',
    save_path: str = 'scores.json'
):
    pass


def score_task_2(
    models_regex: str,
    local_data_dir: str,
    comment: str = '',
    scores_dir: str = 'scores',
    save_path: str = 'scores.json'
):
    pass


def score_task_123(
    models_regex: str,
    local_data_dir: str,
    comment: str = '',
    scores_dir: str = 'scores',
    save_path: str = 'scores.json'
):
    path_to_scorer_script = 'evaluation/semeval2020_06_evaluation_main.py'
    path_to_eval_config = 'evaluation/configs/eval_test.yaml'

    scores = {}
    for part in ['test', 'dev']:
        best_task_1_predictions_regex = \
            f'{models_regex}/{part}_best_sent_type*.tsv'
        best_task_2_predictions_regex = \
            f'{models_regex}/{part}_best_tags_sequence*.tsv'
        best_task_3_predictions_regex = \
            f'{models_regex}/{part}_best_relations_sequence*.tsv'

        print('best task 1 predictions:')
        print(glob(best_task_1_predictions_regex))
        print('best task 2 predictions:')
        print(glob(best_task_2_predictions_regex))
        print('best task 3 predictions:')
        print(glob(best_task_3_predictions_regex))

        for pool_type in ['max_score', 'ellections']:
            scores[f'{part}-best_sent_type-{pool_type}'] = score_task_1_predictions(
                path_to_scorer_script=path_to_scorer_script,
                path_to_gold_data=os.path.join(local_data_dir, f'task_1/{part}'),
                path_to_eval_config=path_to_eval_config,
                predictions_regex=best_task_1_predictions_regex,
                temp_output='temp_output',
                clean_output=True,
                scores_dir=f'{scores_dir}/task_1/{part}-123-{pool_type}-{comment}',
                pool_type=pool_type
            )

            scores[f'{part}-best_tags_sequence-{pool_type}'] = score_task_2_predictions(
                path_to_scorer_script=path_to_scorer_script,
                path_to_gold_data=os.path.join(local_data_dir, f'task_2/{part}'),
                path_to_eval_config=path_to_eval_config,
                predictions_regex=best_task_2_predictions_regex,
                temp_output='temp_output',
                clean_output=True,
                scores_dir=f'{scores_dir}/task_2/{part}-123-{pool_type}-{comment}',
                pool_type=pool_type
            )

    json.dump(scores, open(save_path, 'w'))
    print(scores)


def main(
    tasks: str,
    models_regex: str,
    local_data_dir: str,
    comment: str = '',
    scores_dir: str = 'scores',
    save_path: str = 'scores.json'
):
    tasks = str(tasks)

    assert tasks in ['12', '2', '123']
    if tasks == '123':
        score_task_123(
            models_regex=models_regex,
            local_data_dir=local_data_dir,
            comment=comment,
            scores_dir=scores_dir,
            save_path=save_path
        )
    elif tasks == '12':
        score_task_12(
            models_regex=models_regex,
            local_data_dir=local_data_dir,
            comment=comment,
            scores_dir=scores_dir,
            save_path=save_path
        )
    else:
        score_task_2(
            models_regex=models_regex,
            local_data_dir=local_data_dir,
            comment=comment,
            scores_dir=scores_dir,
            save_path=save_path
        )


if __name__ == '__main__':
    fire.Fire(main)
