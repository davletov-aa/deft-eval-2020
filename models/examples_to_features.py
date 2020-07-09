from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_xlnet import XLNetTokenizer
from transformers.configuration_bert import BertConfig
from transformers.configuration_roberta import RobertaConfig
from transformers.tokenization_roberta import RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
from .multitask_bert import BertForMultitaskLearning
from .multitask_roberta import RobertaForMultitaskLearning
import torch
import os
import json
from collections import Counter


class InputExample(object):

    def __init__(
        self,
        guid: str,
        tokens: list,
        sent_type: str,
        tags_sequence: list,
        relations_sequence: list,
        tags_ids: list,
        sent_start: int,
        sent_end: int,
        subj_start: int,
        subj_end: int,
        start_char: int,
        end_char: int,
        source: str,
        infile_offsets: list
    ):
        self.guid = guid
        self.tokens = tokens
        self.sent_type = sent_type
        self.tags_sequence = tags_sequence
        self.relations_sequence = relations_sequence
        self.tags_ids = tags_ids
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.subj_start = subj_start
        self.subj_end = subj_end
        self.start_char = start_char
        self.end_char = end_char
        self.source = source
        self.infile_offsets = infile_offsets


class DataProcessor(object):
    """Processor for the DEFTEVAL data set."""

    def __init__(
        self,
        filter_task_3: bool = False
    ):
        self.filter_task_3 = filter_task_3

    def _read_json(self, input_file):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
            if self.filter_task_3:
                data = [
                    example for example in data if example['relation_id'] == 0
                ]
        return data

    def get_train_examples(self, data_dir):
        return self.create_examples(
            self._read_json(
                os.path.join(data_dir, f"train.json")
            ),
            "train"
        )

    def get_dev_examples(self, data_dir):
        return self.create_examples(
            self._read_json(
                os.path.join(data_dir, f"dev.json")
            ),
            "dev"
        )

    def get_test_examples(self, test_file):
        return self.create_examples(
            self._read_json(test_file), "test")

    def get_sent_type_labels(self, data_dir, logger=None):
        dataset = self._read_json(os.path.join(data_dir, f"train.json"))
        counter = Counter()
        labels = []
        for example in dataset:
            counter[example['sent_type']] += 1
        if logger is not None:
            logger.info(f"sent_type: {len(counter)} labels")
        for label, counter in counter.most_common():
            if logger is not None:
                logger.info("%s: %.2f%%" % (label, counter * 100.0 / len(dataset)))
            if label not in labels:
                labels.append(label)
        return labels

    def get_sequence_labels(
        self,
        data_dir: str,
        sequence_type: str = 'tags_sequence',
        logger = None
    ):
        dataset = self._read_json(
            os.path.join(data_dir, "train.json")
        )
        denominator = len([
            lab for example in dataset for lab in example[sequence_type]
        ])
        counter = Counter()
        labels = []
        for example in dataset:
            for lab in example[sequence_type]:
                counter[lab] += 1
        if logger is not None:
            logger.info(f"{sequence_type}: {len(counter)} labels")

        for label, counter in counter.most_common():
            if logger is not None:
                logger.info("%s: %.2f%%" % (label, counter * 100.0 / denominator))
            if label not in labels:
                labels.append(label)
        return labels


    def create_examples(self, dataset, set_type):
        examples = []
        for example in dataset:
            examples.append(
                InputExample(
                    guid=f"{set_type}-{example['idx']}",
                    tokens=example["tokens"],
                    sent_type=example["sent_type"],
                    tags_sequence=example["tags_sequence"],
                    relations_sequence=example["relations_sequence"],
                    tags_ids=example["tags_ids"],
                    sent_start=example["sent_start"],
                    sent_end=example["sent_end"],
                    subj_start=example["subj_start"],
                    subj_end=example["subj_end"],
                    start_char=example["start_char"],
                    end_char=example["end_char"],
                    source=example["source"],
                    infile_offsets=example["infile_offsets"]
                )
            )
        return examples


def get_dataloader_and_tensors(
        features: list,
        batch_size: int
):
    input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long
    )
    input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long
    )
    segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long
    )
    sent_type_labels_ids = torch.tensor(
        [f.sent_type_id for f in features],
        dtype=torch.long
    )
    tags_sequence_labels_ids = torch.tensor(
        [f.tags_sequence_ids for f in features],
        dtype=torch.long
    )
    relations_sequence_labels_ids = torch.tensor(
        [f.relations_sequence_ids for f in features],
        dtype=torch.long
    )
    token_valid_pos_ids = torch.tensor(
        [f.token_valid_pos_ids for f in features],
        dtype=torch.long
    )
    eval_data = TensorDataset(
        input_ids, input_mask, segment_ids,
        sent_type_labels_ids, tags_sequence_labels_ids, relations_sequence_labels_ids, token_valid_pos_ids
    )

    dataloader = DataLoader(eval_data, batch_size=batch_size)

    return dataloader, sent_type_labels_ids, tags_sequence_labels_ids, relations_sequence_labels_ids

tokenizers = {
    "bert-large-uncased": BertTokenizer,
    "xlnet-large-cased": XLNetTokenizer,
    "roberta-large": RobertaTokenizer
}

models = {
    "bert-large-uncased": BertForMultitaskLearning,
    "roberta-large": RobertaForMultitaskLearning
}

configs = {
    "bert-large-uncased": BertConfig,
    "roberta-large": RobertaConfig
}
