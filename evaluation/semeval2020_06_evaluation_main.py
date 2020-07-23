"""Evaluation for Semeval 2020 Task 06: DeftEval
How to use:
    1. Adjust settings in the .yaml configuration file
    2. Run this script from the command line

Subtask 01 Input files:
    gold_fname: tab-separated .deft files for Subtask 1 with the following columns (no column headers):
        Sentence    Label
        Label can be HasDef or NoDef
    pred_fname: same format as gold_fname above

Subtask 02 Input files:
    gold_fname: tab-separated .deft files for Subtask 2 with the following columns (no column headers):
        token filename token_start token_end label
    pred_fname: same format as gold_fname above

Subtask 03 Input files:
    gold_fname: tab-separated .deft files for Subtask 3 with the following columns (no column headers):
        token filename token_start token_end label tag_id relation_root relation_name
    pred_fname: same format as gold_fname above

Notes:
    1. pred_fname must have the same columns as gold_fname in the same order
    2. all columns in pred_fname must have the same data as in gold_fname, except for the columns being predicted
        - i.e. label for Subtask 2 and relation_root and relation_name for Subtask 3
    3. pred_fname may not contain any relations that did not appear in the training data
"""

from pathlib import Path
from yaml import safe_load
from evaluation_sub1 import task_1_eval_main
from evaluation_sub2 import task_2_eval_main
from evaluation_sub3 import task_3_eval_main
import os
import sys
import fire


def main(config_path, ref_path, res_path, output_dir):
    """Run the evaluation script(s)
    Inputs:
        cfg: configuration dictionary loaded from yaml
    """

    cfg = safe_load(Path(config_path).open())
    ref_path = Path(ref_path)
    res_path = Path(res_path)
    for i in ['1', '2', '3']:
        os.makedirs(output_dir + f'_task_{i}', exist_ok=True)

    eval_task_1 = cfg['task_1']['do_eval']
    eval_task_2 = cfg['task_2']['do_eval']
    eval_task_3 = cfg['task_3']['do_eval']

    task_1_report = '-1'
    task_2_report = '-1'
    task_3_report = '-1'

    if eval_task_1:
        print('task_1_eval_labels:', cfg['task_1']['eval_labels'])
        task_1_report = task_1_eval_main(ref_path, res_path, output_dir + '_task_1', cfg['task_1']['eval_labels'])
        if task_1_report:
            print(task_1_report)
        print()

    if eval_task_2:
        print('task_2_eval_labels:', cfg['task_2']['eval_labels'])
        task_2_report = task_2_eval_main(ref_path, res_path, output_dir + '_task_2', cfg['task_2']['eval_labels'])
        if task_2_report:
            print(task_2_report)
        print()

    if eval_task_3:
        print('task_3_eval_labels:', cfg['task_3']['eval_labels'])
        task_3_report = task_3_eval_main(ref_path, res_path, output_dir + '_task_3', cfg['task_3']['eval_labels'],
                                         cfg['task_3']['eval_labels'])
        if task_3_report:
            print(task_3_report)
        print()

    return {
        "task_1": task_1_report,
        "task_2": task_2_report,
        "task_3": task_3_report,
    }

if __name__ == "__main__":
    fire.Fire(main)
