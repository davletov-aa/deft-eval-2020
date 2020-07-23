import fire
import os
from utils.data_processing import *
from glob import glob


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
    suffix = {
        1: 'best_sent_type',
        2: 'best_tags_sequence',
        3: 'best_relations_sequence'
    }
    for task_id in [1, 2, 3]:
        for part in ['test', 'dev']:
            best_task_predictions_regex = \
                f'{models_regex}/{part}_{suffix[task_id]}*.tsv'
            print(f'best task {task_id} predictions:')
            print(glob(best_task_predictions_regex))

            for i, pool_type in enumerate(['max_score', 'ellections']):
                if task_id == 3 and i == 1:
                    continue

                scores[f'{part}-{suffix[task_id]}-{pool_type}'] = score_tasks_predictions(
                    path_to_scorer_script=path_to_scorer_script,
                    path_to_gold_data=os.path.join(local_data_dir, f'task_{task_id}/{part}'),
                    path_to_eval_config=path_to_eval_config,
                    predictions_regex=best_task_predictions_regex,
                    temp_output='temp_output',
                    clean_output=True,
                    scores_dir=f'{scores_dir}/task_{task_id}/{part}-123-{pool_type}-{comment}',
                    pool_type=pool_type,
                    task_id=task_id
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
