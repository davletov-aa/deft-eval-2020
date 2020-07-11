import os
import pandas as pd
import re
from collections import defaultdict
from itertools import groupby
from tqdm import tqdm
import json


EVAL_RELATIONS = [
    'Direct-Defines', 'Indirect-Defines', 'AKA', 'Refers-To', 'Supplements'
]


EVAL_TAGS = [
    'B-Term', 'I-Term', 'B-Definition', 'I-Definition',
    'B-Alias-Term', 'I-Alias-Term', 'B-Referential-Definition', 'I-Referential-Definition',
    'B-Referential-Term', 'I-Referential-Term', 'B-Qualifier', 'I-Qualifier'
]


def create_local_dataset_folder(
    task_converter_path: str,
    source_folder: str,
    output_folder: str,
    test_folder: str
):
    if os.path.exists(output_folder):
        os.system(f'rm -r {output_folder}/*')
    else:
        os.makedirs(output_folder, exist_ok=True)

    # create dataset structure
    for i in range(1, 4):
        for part in ['train', 'dev']:
            os.makedirs(
                os.path.join(output_folder, f'task_{i}', part),
                exist_ok=True
            )

    # construct dataset files according our structure
    for part in ['train', 'dev']:
        for i in range(2, 4):
            out = os.path.join(output_folder, f'task_{i}')
            os.system(f"cp -r {source_folder}/{part} {out}")

        out = os.path.join(output_folder, 'task_1', part)
        os.system(
            f"python {task_converter_path} {source_folder}/{part}/ {out}"
        )

    # rename all files
    for i in range(2, 4):
        for part in ['train', 'dev']:
            files = [file for file in os.listdir(f'{output_folder}/task_{i}/{part}')]

            for file in files:
                new_file = f'task_{i}_' + file
                infile = os.path.join(output_folder, f'task_{i}', part, file)
                outfile = os.path.join(output_folder, f'task_{i}', part, new_file)
                os.system(f'mv {infile} {outfile}')

    # same for test files
    for i in range(1, 4):
        dest_folder = os.path.join(output_folder, f'task_{i}', 'test')
        folder = os.path.join(test_folder, f'subtask_{i}')

        if os.path.exists(dest_folder):
            os.system(f"rm -r {dest_folder}/*")
        else:
            os.makedirs(dest_folder, exist_ok=True)

        if os.path.exists(folder):
            files = os.listdir(folder)

            for file in files:
                infile = os.path.join(folder, file)
                outfile = os.path.join(dest_folder, file)
                os.system(f'cp {infile} {outfile}')


def read_dataset(
    data_dir: str,
    task_id: int = 1,
    sep: str = '\t',
    test_folder_name: str = 'test',
    do_filter: bool = True,
    check_if_correct: bool = True
):
    print(EVAL_RELATIONS)
    print(EVAL_TAGS)
    assert task_id in [1, 2, 3]
    dataset = pd.DataFrame()
    for part in ['train', 'dev', test_folder_name]:
        dataset_part = read_part(os.path.join(data_dir, f'task_{task_id}', part),
                                 task_id, sep, check_if_correct, do_filter)
        dataset_part.loc[:, 'part'] = part
        dataset = dataset.append(dataset_part, ignore_index=True)
    return dataset


def read_part(
    data_part_dir: str,
    task_id: int,
    sep: str = '\t',
    check_if_correct: bool = True,
    do_filter: bool = True
):
    reader = {
        1: read_task_1,
        2: read_task_2_or_3,
        3: read_task_2_or_3
    }
    part = reader[task_id](
        data_dir=data_part_dir, sep=sep, do_filter=do_filter, task_id=task_id
    )

    if task_id == 2:
        columns = [x for x in part.keys()]
        task_2_columns = [
            'token', 'source', 'start_char',
            'end_char', 'tag', 'infile_offsets',
            'part'
        ]
        for column in columns:
            if column not in task_2_columns:
                del part[column]

    dataset = pd.DataFrame(part)

    if task_id == 3 and not hasattr(dataset, 'tag_id'):
        dataset = task_2_to_task_3(dataset)

    if task_id == 1:
        dataset.loc[:, 'orig_sent'] = dataset.token.copy().apply(lambda x: ' '.join(x))
        dataset.loc[:, 'token'] = dataset.token.apply(lambda x: ' '.join(x).strip()).apply(lambda x: x.split(' '))

    if task_id == 3 and check_if_correct:
        dataset.loc[:, 'is_correct'] = dataset.apply(lambda r: is_correct(r.root_id, r.tag_id), axis=1)
        print(f'{len(dataset) - sum(dataset.is_correct.values)} non correct examples detected!')
    return dataset


def task_2_to_task_3(
    task_2_dataset: pd.DataFrame
):
    new_df = task_2_dataset.copy()
    new_df.loc[:, 'tag_id'] = new_df.token.apply(lambda x: ['-1'] * len(x))
    new_df.loc[:, 'root_id'] = new_df.token.apply(lambda x: ['-1'] * len(x))
    new_df.loc[:, 'relation'] = new_df.token.apply(lambda x: ['0'] * len(x))

    return new_df


def read_task_1(
    data_dir: str,
    sep: str = '\t',
    columns: str = '',
    do_filter: bool = False,
    task_id: int = 1
):
    X, y, source = [], [], []
    for file in sorted(os.listdir(data_dir)):
        print(file)
        with open(os.path.join(data_dir, file)) as fp:
            file_lines = fp.readlines()

        if len(file_lines[0].split(sep)) == 1:
            X.extend([line.rstrip().split(' ') for line in file_lines])
            y.extend(['0'] * len(X))
        else:
            sents, labels = zip(*[line.strip().split(sep) for line in file_lines])
            y.extend([ex.strip('"') for ex in labels])
            X.extend([ex.strip('"').split(' ') for ex in sents])

        source.extend([str(file)] * len(file_lines))
    result = {'token': X, 'label': y, 'source': source}
    return result

def read_task_2_or_3(
    data_dir: str,
    sep: str = '\t',
    columns: str = '+'.join(
        [
            'token', 'source', 'start_char',
            'end_char', 'tag', 'tag_id',
            'root_id', 'relation'
        ]
    ),
    task_id: int = 3,
    do_filter: bool = True
):
    assert task_id in [2, 3], f'task_id should be from [1, 2]'
    columns = columns.split('+')

    result = defaultdict(list)
    sentences = defaultdict(list)
    num_columns = None

    for file in sorted(os.listdir(data_dir)):
        print(file)
        with open(os.path.join(data_dir, file)) as fp:
            file_lines = [line.strip() for line in fp.readlines()]

        infile_offset = 0
        if num_columns is None:
            num_columns = len(file_lines[0].split(sep))

        for is_sep, lines in groupby(file_lines, key=lambda line: line == ''):
            lines = list(lines)
            if not is_sep:
                sentences[file].append((lines, [i + infile_offset for i in range(len(lines))]))
            infile_offset += len(lines)

    all_sentences = \
    [
        [
            [column.strip() for column in token.split(sep)] + [infile_offset]
            for token, infile_offset in zip(sentence, infile_offsets)
        ]
        for file_sentences in sentences.values()
        for (sentence, infile_offsets) in file_sentences
    ]

    sentence_starts = \
    [
        i for i, sentence in enumerate(all_sentences)
        if len(sentence) == 2 and sentence[0][0].isdigit() and (sentence[1][0] == '.')
    ]

    window_sentences = []
    for i in range(len(sentence_starts) - 1):
        window_sentence = []
        for j in range(sentence_starts[i], sentence_starts[i + 1]):
            window_sentence.extend(all_sentences[j])
        window_sentences.append(window_sentence)

    last_window_sentence = []
    for j in range(sentence_starts[-1], len(all_sentences)):
        last_window_sentence.extend(all_sentences[j])
    window_sentences.append(last_window_sentence)

    columns = columns[:num_columns]

    def filter_value(value, eval_labels, neg_label):
        if value.strip() not in eval_labels and value.strip() != neg_label:
            value = neg_label
        return value

    for window_sentence in window_sentences:
        sentence = defaultdict(list)
        for token in window_sentence:
            for i, column_name in enumerate(columns):
                column_value = token[i]
                if task_id == 3 and column_name == 'relation':
                    column_value = filter_value(
                        column_value, EVAL_RELATIONS, '0'
                    )

                if task_id == 2 and column_name == 'tag':
                    column_value = filter_value(
                        column_value, EVAL_TAGS, 'O'
                    )

                sentence[column_name].append(token[i])
            sentence['infile_offsets'].append(token[-1])

        for column_name in sentence:
            result[column_name].append(sentence[column_name])

    return result


def is_correct(root_id, tag_id):
    answer = False
    pairs_set = set([(from_, to_) for from_, to_ in zip(root_id, tag_id)])

    subjects_in_relations = \
        set([
            from_ for (from_, to_) in pairs_set
            if from_ not in ['-1', '0'] and to_ != '-1'
        ])
    roots = set([to_ for (from_, to_) in pairs_set if from_ == '0'])

    if all([subj in subjects_in_relations for subj in roots]):
        answer = True
    return answer


def create_multitask_dataset(
    dataset_path: str,
    output_dir: str,
    do_filter=True,
    filter_non_correct=True
):
    dataset = read_dataset(dataset_path, task_id=3, do_filter=do_filter, check_if_correct=True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for from_part in ['train', 'dev', 'test']:
        task_3_dataset = dataset[dataset.part == from_part].copy()

        if filter_non_correct:
            task_3_dataset = task_3_dataset[task_3_dataset.is_correct == True]

        task_3_dataset = task_3_dataset.reset_index()

        source_for_task_1_dataset = os.path.join(dataset_path, f'task_3/{from_part}/')
        examples = create_multitask_examples(task_3_dataset, source_for_task_1_dataset)

        output_file = os.path.join(output_dir, f'{from_part}.json')
        json.dump(examples, open(output_file, 'w'))


def create_multitask_examples(
    task_3_dataset: pd.DataFrame,
    source_for_task_1_dataset: str
):
    task_1_dataset = get_task_1_sentences_official_way(source_for_task_1_dataset)
    sent_starts, sent_ends, sent_types, sent_ids = \
        get_sent_starts_sent_ends_and_sent_types(task_1_dataset, task_3_dataset)

    roots, relations = get_roots_and_relations(task_3_dataset)

    df = task_3_dataset.copy()

    if not hasattr(df, 'tag_id'):
        df = task_2_to_task_3(df)

    df.loc[:, 'sent_starts'] = sent_starts
    df.loc[:, 'sent_ends'] = sent_ends
    df.loc[:, 'sent_type_labels'] = sent_types
    df.loc[:, 'roots'] = roots
    df.loc[:, 'relations'] = relations

    examples = []
    for row_id, row in enumerate(df.itertuples()):
        token = row.token
        tag = row.tag
        tag_id = row.tag_id
        infile_offsets = row.infile_offsets
        start_char = row.start_char
        end_char = row.end_char
        source = row.source[0]

        for sent_id, (sent_start, sent_end) in enumerate(zip(row.sent_starts, row.sent_ends)):
            sent_type_label = row.sent_type_labels[sent_id]
            for relation_id, ((subj_start, subj_end), relation) in enumerate(zip(row.roots, row.relations)):
                example = {
                    'idx': f'{row_id}-{sent_id}-{subj_id}-{subj_start}+{subj_end}',
                    'tokens': token,
                    'sent_start': sent_start,
                    'sent_end': sent_end,
                    'sent_type': sent_type_label,
                    'tags_sequence': tag[:len(token)],
                    'subj_start': subj_start,
                    'subj_end': subj_end,
                    'relations_sequence': relation,
                    'tags_ids': tag_id[:len(token)],
                    'infile_offsets': infile_offsets,
                    'source': source,
                    'start_char': start_char,
                    'end_char': end_char,
                    'subj_id': relation_id,
                    'sent_id': sent_id
                }
                examples.append(example)

    return examples


def get_sent_starts_sent_ends_and_sent_types(task_1_dataset, task_3_dataset):
    total_sent_starts, total_sent_ends, total_sent_types = [], [], []
    total_ids = []

    for row in tqdm(task_3_dataset.itertuples(), total=len(task_3_dataset)):
        sent_starts, sent_ends, sent_types, sent_ids = [], [], [], []
        source = 'task_1_' + row.source[0].split('/')[-1]
        single_sentences = [
            sent for sent in task_1_dataset[task_1_dataset.source == source].token
        ]
        single_sentences_types = [
                sent_type for sent_type
                in task_1_dataset[task_1_dataset.source == source].label
        ]
        single_sentence_ids = [idx for idx in task_1_dataset[task_1_dataset.source == source].idx.values]
        window_sentence = row.token

        for sent_id, single_sentence in enumerate(single_sentences):
            for single_sentence_start in range(len(window_sentence) - len(single_sentence) + 1):
                if window_sentence[single_sentence_start:
                single_sentence_start + len(single_sentence)] == single_sentence:
                    sent_starts.append(single_sentence_start)
                    sent_ends.append(single_sentence_start + len(single_sentence) - 1)
                    sent_types.append(single_sentences_types[sent_id])
                    sent_ids.append(single_sentence_ids[sent_id])
                    break
        total_sent_starts.append(sent_starts)
        total_sent_ends.append(sent_ends)
        total_sent_types.append(sent_types)
        total_ids.append(sent_ids)

    return total_sent_starts, total_sent_ends, total_sent_types, total_ids


def get_roots_and_relations(task_3_dataset):
    assert hasattr(task_3_dataset, 'tag'), "dataset must have \"tag\" column"

    def get_relations_for_subjects(subjects, tag_ids, root_ids, relations):
        relations_for_subjects = []
        for (subj_start, subj_end) in subjects:
            relation = []
            cur_tag_id = tag_ids[subj_start]
            for rel, tag_id, root_id in zip(relations, tag_ids, root_ids):
                if root_id == cur_tag_id:
                    relation.append(rel)
                else:
                    relation.append('0')
            relations_for_subjects.append(relation)
        return relations_for_subjects

    total_subjects, total_relations = [], []
    is_test_dataset = not hasattr(task_3_dataset, 'relation')

    for row in task_3_dataset.itertuples():
        tags = row.tag
        tag_starts = [i for i, tag in enumerate(tags) if tag.startswith('B-')]
        additional_starts = []
        if tags[0].startswith('I-'):
            additional_starts.append(i)
        for i in range(len(tags) - 1):
            if tags[i] == 'O' and tags[i + 1].startswith('I-'):
                additional_starts.append(i + 1)

        tag_starts = tag_starts + additional_starts
        tag_ends = []

        for tag_start in tag_starts:
            position = tag_start + 1
            while position < len(tags) and tags[position].startswith('I-'):
                position += 1
            tag_ends.append(position)

        subjects = [(subj_start, subj_end - 1) for subj_start, subj_end in zip(tag_starts, tag_ends)]
        if is_test_dataset:
            relations = []
            for _ in subjects:
                relations.append(['0'] * len(tags))
        else:
            relations = get_relations_for_subjects(
                subjects=subjects, tag_ids=row.tag_id,
                root_ids=row.root_id, relations=row.relation
            )

        total_subjects.append(subjects)
        total_relations.append(relations)

    return total_subjects, total_relations


def get_task_1_sentences_official_way(
    source_dir: str
):
    sentences = pd.DataFrame(columns=['token', 'label', 'infile_offsets', 'source', 'idx'])
    source_files = [file for file in os.listdir(source_dir) if file.endswith('.deft')]

    for source_file in source_files:
        infile_offsets = []
        num_sents = 0
        with open(os.path.join(source_dir, source_file)) as source_text:

            has_def = 0
            new_sentence = ''
            all_lines = list(source_text.readlines())

            for index, line in enumerate(all_lines):
                if re.match('^\s+$', line) \
                    and len(new_sentence) > 0 \
                    and (
                        not re.match(r'^\s*\d+\s*\.$', new_sentence)
                        or all_lines[index - 1] == '\n'
                    ):
                    num_sents += 1
                    sentences = sentences.append({
                        'token': new_sentence.lstrip().split(' '), 'label': has_def,
                        'infile_offsets': infile_offsets,
                        'idx': num_sents,
                        "source": f"task_1_{source_file[7:]}"}, ignore_index=True)

                    new_sentence = ''
                    has_def = 0
                    infile_offsets = []

                if line == '\n':
                    continue

                line_parts = line.split('\t')
                new_sentence = new_sentence + ' ' + line_parts[0]
                infile_offsets.append(index)

                if len(line_parts) > 4 and line_parts[4][3:] == 'Definition':
                    has_def = 1
            if len(new_sentence) > 0:
                num_sents += 1
                sentences = sentences.append(
                    {
                        'token': new_sentence.lstrip().split(' '), 'label': has_def,
                        'infile_offsets': infile_offsets,
                        'idx': num_sents,
                        "source": f"task_1_{source_file[7:]}"
                    },
                    ignore_index=True
                )
    return sentences

