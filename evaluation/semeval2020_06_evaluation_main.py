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


EVAL_RELATIONS = [
    'Direct-Defines', 'Indirect-Defines', 'AKA', 'Refers-To', 'Supplements'
]


EVAL_TAGS = [
    'B-Term', 'I-Term', 'B-Definition', 'I-Definition',
    'B-Alias-Term', 'I-Alias-Term', 'B-Referential-Definition', 'I-Referential-Definition',
    'B-Referential-Term', 'I-Referential-Term', 'B-Qualifier', 'I-Qualifier'
]

EVAL_SENT_TYPES = [
    '0', '1'
]


def main(
    ref_path, res_path, output_dir,
    eval_task_1: str = "false",
    eval_task_2: str = "false",
    eval_task_3: str = "false"
):
    """Run the evaluation script(s)
    """

    ref_path = Path(ref_path)
    res_path = Path(res_path)
    eval_task_1 = eval_task_1.lower() == 'true'
    eval_task_2 = eval_task_2.lower() == 'true'
    eval_task_3 = eval_task_3.lower() == 'true'

    for eval_task_id, eval_task in enumerate([eval_task_1, eval_task_2, eval_task_3], start=1):
        if eval_task:
            os.makedirs(output_dir + f'_task_{eval_task_id}', exist_ok=True)


    task_1_report = '-1'
    task_2_report = '-1'
    task_3_report = '-1'

    if eval_task_1:
        print('task_1_eval_labels:', EVAL_SENT_TYPES)
        task_1_report = task_1_eval_main(ref_path, res_path, output_dir + '_task_1', EVAL_SENT_TYPES)
        if task_1_report:
            print(task_1_report)
        print()

    if eval_task_2:
        print('task_2_eval_labels:', EVAL_TAGS)
        task_2_report = task_2_eval_main(ref_path, res_path, output_dir + '_task_2', EVAL_TAGS)
        if task_2_report:
            print(task_2_report)
        print()

    if eval_task_3:
        print('task_3_eval_labels:', EVAL_RELATIONS)
        task_3_report = task_3_eval_main(ref_path, res_path, output_dir + '_task_3', EVAL_RELATIONS,
                                         EVAL_TAGS)
        if task_3_report:
            print(task_3_report)
        print()

    # return {
    #     "task_1": task_1_report,
    #     "task_2": task_2_report,
    #     "task_3": task_3_report,
    # }

if __name__ == "__main__":
    fire.Fire(main)
