import argparse
import logging
import os
import random
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from torch.nn import CrossEntropyLoss

from transformers.optimization import (
    AdamW, get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from transformers.file_utils import (
    PYTORCH_PRETRAINED_BERT_CACHE,
    WEIGHTS_NAME, CONFIG_NAME
)

from tqdm import tqdm
from models.examples_to_features import (
    get_dataloader_and_tensors,
    models, tokenizers, DataProcessor, configs
)
from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support, classification_report
)
from torch.nn import CrossEntropyLoss
from utils.data_processing import (
    EVAL_TAGS, EVAL_RELATIONS
)

from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
eval_logger = logging.getLogger("__scores__")


def compute_all_metrics(
    sent_type_labels, sent_type_preds,
    tags_sequence_labels, tags_sequence_preds,
    relations_sequence_labels, relations_sequence_preds,
    label2id, loss_info=None, logger=None
):
    eval_tags_sequence_labels = [
        (label2id['tags_sequence'][lab]) for lab in EVAL_TAGS
    ]
    eval_relations_sequence_labels = [
        (label2id['relations_sequence'][lab]) for lab in EVAL_RELATIONS
    ]

    task_1_report = classification_report(
        sent_type_labels, sent_type_preds, labels=[0, 1], output_dict=True
    )
    task_2_report = classification_report(
        tags_sequence_labels, tags_sequence_preds,
        labels=eval_tags_sequence_labels, output_dict=True
    )
    task_3_report = classification_report(
        relations_sequence_labels, relations_sequence_preds,
        labels=eval_relations_sequence_labels, output_dict=True
    )

    result = {}
    for x in ['0', '1', 'weighted avg', 'macro avg']:
        for metrics in ['precision', 'recall', 'f1-score', 'support']:
            result[f"sent_type_{x.replace(' ', '-')}_{metrics}"] = \
                round(task_1_report[x][metrics], 6)

    id2label = {
        val: key for key, val in label2id['tags_sequence'].items()
    }
    id2label['weighted avg'] = 'weighted-avg'
    id2label['macro avg'] = 'macro-avg'
    for x in eval_tags_sequence_labels + ['weighted avg', 'macro avg']:
        for metrics in ['precision', 'recall', 'f1-score', 'support']:
            result[f"tags_sequence_{id2label[x]}_{metrics}"] = \
                round(task_2_report[str(x)][metrics], 6)

    id2label = {
        val: key for key, val in label2id['relations_sequence'].items()
    }
    id2label['weighted avg'] = 'weighted-avg'
    id2label['macro avg'] = 'macro-avg'
    for x in eval_relations_sequence_labels + ['weighted avg', 'macro avg']:
        for metrics in ['precision', 'recall', 'f1-score', 'support']:
            result[f"relations_sequence_{id2label[x]}_{metrics}"] = \
                round(task_3_report[str(x)][metrics], 6)
    if logger is not None:
        logger.info("=====================================")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        if loss_info is not None:
            for key in sorted(loss_info.keys()):
                logger.info(
                    "  %s = %s", key, str(loss_info[key])
                )

    return result


def evaluate(
        model, device, eval_dataloader,
        eval_sent_type_labels_ids,
        eval_tags_sequence_labels_ids,
        eval_relations_sequence_labels_ids,
        label2id,
        compute_metrics=True,
        verbose=False, cur_train_mean_loss=None,
        logger=None,
        skip_every_n_examples=1
    ):
    model.eval()

    num_sent_type_labels = model.num_sent_type_labels
    num_tags_sequence_labels = model.num_tags_sequence_labels
    num_relations_sequence_labels = model.num_relations_sequence_labels

    sent_type_clf_weight = model.sent_type_clf_weight
    tags_sequence_clf_weight = model.tags_sequence_clf_weight
    relations_sequence_clf_weight = model.relations_sequence_clf_weight


    eval_loss = defaultdict(float)
    nb_eval_steps = 0
    preds = defaultdict(list)

    for batch_id, batch in enumerate(tqdm(
            eval_dataloader, total=len(eval_dataloader),
            desc='validation ... '
        )):

        if skip_every_n_examples != 1 and (batch_id + 1) % skip_every_n_examples != 1:
            continue

        batch = tuple([elem.to(device) for elem in batch])

        input_ids, input_mask, segment_ids, \
            sent_type_labels_ids, tags_sequence_labels_ids, \
            relations_sequence_labels_ids, token_valid_pos_ids = batch

        with torch.no_grad():
            outputs, loss = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                sent_type_labels=sent_type_labels_ids,
                tags_sequence_labels=tags_sequence_labels_ids,
                relations_sequence_labels=relations_sequence_labels_ids,
                token_valid_pos_ids=token_valid_pos_ids,
                device=device
            )

        sent_type_logits, tags_sequence_logits, \
            relations_sequence_logits = outputs[:3]

        if compute_metrics:
            eval_loss['sent_type_loss'] += \
                loss['sent_type_loss'].mean().item()
            eval_loss['tags_sequence_loss'] += \
                loss['tags_sequence_loss'].mean().item()
            eval_loss['relations_sequence_loss'] += \
                loss['relations_sequence_loss'].mean().item()
            eval_loss['weighted_loss'] += \
                loss['weighted_loss'].mean().item()

        nb_eval_steps += 1
        preds['sent_type'].append(
            sent_type_logits.detach().cpu().numpy()
        )
        preds['tags_sequence'].append(
            tags_sequence_logits.detach().cpu().numpy()
        )
        preds['relations_sequence'].append(
            relations_sequence_logits.detach().cpu().numpy()
        )

    preds['sent_type'] = np.concatenate(preds['sent_type'], axis=0)
    preds['tags_sequence'] = np.concatenate(preds['tags_sequence'], axis=0)
    preds['relations_sequence'] = np.concatenate(
        preds['relations_sequence'],
        axis=0
    )

    scores = {}

    for key in preds:
        scores[key] = softmax(preds[key], axis=-1).max(axis=-1)
        preds[key] = preds[key].argmax(axis=-1)

    if compute_metrics:
        for key in eval_loss:
            eval_loss[key] = eval_loss[key] / nb_eval_steps
        if cur_train_mean_loss is not None:
            eval_loss.update(cur_train_mean_loss)

        result = compute_all_metrics(
            eval_sent_type_labels_ids.numpy(), preds['sent_type'],
            np.array([x for y in eval_tags_sequence_labels_ids.numpy() for x in y]),
            np.array([x for y in preds['tags_sequence'] for x in y]),
            np.array([x for y in eval_relations_sequence_labels_ids.numpy() for x in y]),
            np.array([x for y in preds['relations_sequence'] for x in y]),
            label2id, loss_info=eval_loss,
            logger=logger
        )
    else:
        result = {}

    for key in eval_loss:
        result[key] = eval_loss[key]

    model.train()

    return preds, result, scores


def main(args):

    # only for heatmap
    if args.sent_type_clf_weight < 1 and args.relations_sequence_clf_weight < 1:
        print(f'skipping ... {args.output_dir}: both tasks 1 and 3 weights below 1.0')
        return
    if os.path.exists(args.output_dir):
        from glob import glob
        tsv_files = glob(f'{args.output_dir}', '*best*tsv')
        if tsv_files:
            print('already computed: skipping')
        else:
            # os.system(f'rm -r {args.output_dir}')
            print(args.output_dir)

    assert args.context_mode in ['full', 'center', 'left', 'right']

    # if args.output_dirs_to_exclude != '':
    #     output_dirs_to_exclue = json.load(open(args.output_dirs_to_exclude))
    # else:
    #     output_dirs_to_exclue = []

    # if args.output_dir in output_dirs_to_exclue:
    #     print(f'skipping ... {args.output_dir}: from exclude output dirs')
    #     return 0

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "gradient_accumulation_steps parameter should be >= 1"
        )

    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if args.do_train:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True."
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    elif args.do_train or args.do_validate:
        raise ValueError(args.output_dir, 'output_dir already exists')

    suffix = datetime.now().isoformat().replace('-', '_').replace(
        ':', '_').split('.')[0].replace('T', '-')
    if args.do_train:

        train_writer = SummaryWriter(
            log_dir=os.path.join(
                'tensorboard', args.output_dir, 'train'
            )
        )
        dev_writer = SummaryWriter(
            log_dir=os.path.join(
                'tensorboard', args.output_dir, 'dev'
            )
        )

        logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"train_{suffix}.log"), 'w')
        )
        eval_logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"scores_{suffix}.log"), 'w')
        )
    else:
        logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, f"eval_{suffix}.log"), 'w')
        )

    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    processor = DataProcessor(
        filter_task_1=args.filter_task_1,
        filter_task_3=args.filter_task_3
    )

    eval_metrics = {
        eval_metric: True for eval_metric in args.eval_metrics.split('+')
    }

    if args.filter_task_1 and args.do_train:
        assert args.sent_type_clf_weight == 0.0
        eval_metrics.pop('sent_type_1_f1-score')

    if args.filter_task_3 and args.do_train:
        assert args.relations_sequence_clf_weight == 0.0
        eval_metrics.pop('relations_sequence_macro-avg_f1-score')

    if args.sent_type_clf_weight == 0.0 and \
        'sent_type_1_f1-score' in eval_metrics:
        eval_metrics.pop('sent_type_1_f1-score')

    if args.tags_sequence_clf_weight == 0.0 and \
        'tags_sequence_macro-avg_f1-score' in eval_metrics:
        eval_metrics.pop('tags_sequence_macro-avg_f1-score')

    if args.relations_sequence_clf_weight == 0.0 and \
        'relations_sequence_macro-avg_f1-score' in eval_metrics:
        eval_metrics.pop('relations_sequence_macro-avg_f1-score')

    assert len(eval_metrics) > 0, "inconsistent train params"

    if args.context_mode != 'full':
        keys = list(eval_metrics.keys())
        for key in keys:
            if key != 'sent_type_1_f1-score':
                eval_metrics.pop(key)
        assert 'sent_type_1_f1-score' in eval_metrics

    sent_type_labels_list = \
        processor.get_sent_type_labels(args.data_dir, logger)
    tags_sequence_labels_list = \
        processor.get_sequence_labels(
            args.data_dir, logger=logger, sequence_type='tags_sequence'
        )
    relations_sequence_labels_list = \
        processor.get_sequence_labels(
            args.data_dir, logger=logger, sequence_type='relations_sequence'
        )

    label2id = {
        'sent_type': {
            label: i for i, label in enumerate(sent_type_labels_list)
        },
        'tags_sequence': {
            label: i for i, label in enumerate(tags_sequence_labels_list, 1)
        },
        'relations_sequence': {
            label: i for i, label in enumerate(relations_sequence_labels_list, 1)
        }
     }

    id2label = {
        'sent_type': {
            i: label for i, label in enumerate(sent_type_labels_list)
        },
        'tags_sequence': {
            i: label for i, label in enumerate(tags_sequence_labels_list, 1)
        },
        'relations_sequence': {
            i: label for i, label in enumerate(relations_sequence_labels_list, 1)
        }
    }

    num_sent_type_labels = len(sent_type_labels_list)
    num_tags_sequence_labels = len(tags_sequence_labels_list) + 1
    num_relations_sequence_labels = len(relations_sequence_labels_list) + 1

    do_lower_case = 'uncased' in args.model
    tokenizer = tokenizers[args.model].from_pretrained(
        args.model, do_lower_case=do_lower_case
    )

    model_name = args.model

    if args.do_train:
        config = configs[args.model]
        config = config.from_pretrained(
            args.model,
            hidden_dropout_prob=args.dropout
        )
        model = models[model_name].from_pretrained(
            args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE),
            num_sent_type_labels=num_sent_type_labels,
            num_tags_sequence_labels=num_tags_sequence_labels,
            num_relations_sequence_labels=num_relations_sequence_labels,
            sent_type_clf_weight=args.sent_type_clf_weight,
            tags_sequence_clf_weight=args.tags_sequence_clf_weight,
            relations_sequence_clf_weight=args.relations_sequence_clf_weight,
            pooling_type=args.subtokens_pooling_type,
            config=config
        )
        print(
            "task weights:",
            model.sent_type_clf_weight,
            model.tags_sequence_clf_weight,
            model.relations_sequence_clf_weight
        )

    else:
        model = models[model_name].from_pretrained(
            args.output_dir,
            num_sent_type_labels=num_sent_type_labels,
            num_tags_sequence_labels=num_tags_sequence_labels,
            num_relations_sequence_labels=num_relations_sequence_labels,
            sent_type_clf_weight=args.sent_type_clf_weight,
            tags_sequence_clf_weight=args.tags_sequence_clf_weight,
            relations_sequence_clf_weight=args.relations_sequence_clf_weight,
            pooling_type=args.subtokens_pooling_type
        )

    model.to(device)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features, eval_new_examples = model.convert_examples_to_features(
        eval_examples, label2id, args.max_seq_length,
        tokenizer, logger, args.sequence_mode, context_mode=args.context_mode
    )
    logger.info("***** Dev *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_dataloader, eval_sent_type_labels_ids, \
    eval_tags_sequence_labels_ids, eval_relations_sequence_labels_ids = \
        get_dataloader_and_tensors(eval_features, args.eval_batch_size)

    test_file = os.path.join(
        args.data_dir, 'test.json'
    ) if args.test_file == '' else args.test_file
    test_examples = processor.get_test_examples(test_file)

    test_features, test_new_examples = model.convert_examples_to_features(
        test_examples, label2id, args.max_seq_length,
        tokenizer, logger, args.sequence_mode, context_mode=args.context_mode
    )
    logger.info("***** Test *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    test_dataloader, test_sent_type_labels_ids, \
    test_tags_sequence_labels_ids, test_relations_sequence_labels_ids = \
        get_dataloader_and_tensors(test_features, args.eval_batch_size)

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features, _ = model.convert_examples_to_features(
            train_examples, label2id,
            args.max_seq_length, tokenizer, logger, args.sequence_mode,
            context_mode=args.context_mode
        )

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(
                train_features, key=lambda f: np.sum(f.input_mask)
            )
        else:
            random.shuffle(train_features)

        train_dataloader, sent_type_ids, tags_sequence_ids, \
        relations_sequence_ids = \
            get_dataloader_and_tensors(train_features, args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * \
                args.num_train_epochs

        warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = defaultdict(float)
        for eval_metric in eval_metrics:
            best_result[eval_metric] = args.threshold
            if eval_metric.startswith('sent_type'):
                best_result[eval_metric] += 0.2
        print('best results thresholds:')
        print(best_result)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        lr = float(args.learning_rate)

        # if n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    param for name, param in param_optimizer
                    if not any(nd in name for nd in no_decay)
                ],
                'weight_decay': float(args.weight_decay)
            },
            {
                'params': [
                    param for name, param in param_optimizer
                    if any(nd in name for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=lr
        )
        if args.lr_schedule == 'constant_warmup':
            print('lr schedule = constant_warmup')
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_train_optimization_steps
            )

        start_time = time.time()
        global_step = 0

        for epoch in range(1, 1 + int(args.num_train_epochs)):
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            cur_train_loss = defaultdict(float)

            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)

            for step, batch in enumerate(
                tqdm(
                    train_batches, total=len(train_batches),
                    desc='training ... '
                )
            ):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, \
                sent_type_labels_ids, tags_sequence_labels_ids, \
                relations_sequence_labels_ids, token_valid_pos_ids = batch
                train_loss = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    sent_type_labels=sent_type_labels_ids,
                    tags_sequence_labels=tags_sequence_labels_ids,
                    relations_sequence_labels=relations_sequence_labels_ids,
                    token_valid_pos_ids=token_valid_pos_ids,
                    return_outputs=False,
                    device=device
                )
                for key in train_loss:
                    cur_train_loss[key] += train_loss[key].mean().item()

                loss_to_optimize = train_loss['weighted_loss']
                if n_gpu > 1:
                    loss_to_optimize = loss_to_optimize.mean()

                if args.gradient_accumulation_steps > 1:
                    loss_to_optimize = \
                        loss_to_optimize / args.gradient_accumulation_steps

                loss_to_optimize.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm
                )

                tr_loss += loss_to_optimize.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                if args.do_validate and (step + 1) % eval_step == 0:
                    logger.info(
                        'Ep: {}, Stp: {}/{}, usd_t={:.2f}s, loss={:.6f}'.format(
                            epoch, step + 1, len(train_batches),
                            time.time() - start_time, tr_loss / nb_tr_steps
                        )
                    )
                    predict_for_metrics = []
                    cur_train_mean_loss = {}
                    for key in cur_train_loss:
                        cur_train_mean_loss[f'train_{key}'] = \
                            cur_train_loss[key] / nb_tr_steps

                    preds, result, scores = evaluate(
                        model, device, eval_dataloader,
                        eval_sent_type_labels_ids,
                        eval_tags_sequence_labels_ids,
                        eval_relations_sequence_labels_ids,
                        label2id, cur_train_mean_loss=cur_train_mean_loss,
                        logger=eval_logger
                    )

                    result['global_step'] = global_step
                    result['epoch'] = epoch
                    result['learning_rate'] = lr
                    result['batch_size'] = \
                    args.train_batch_size * args.gradient_accumulation_steps

                    for key, value in result.items():
                        dev_writer.add_scalar(key, value, global_step)
                    for key,value in cur_train_mean_loss.items():
                        train_writer.add_scalar(
                            f'running_train_{key}', value, global_step
                        )

                    logger.info("First 20 predictions:")
                    for sent_type_pred, sent_type_label in zip(
                            preds['sent_type'][:20],
                            eval_sent_type_labels_ids.numpy()[:20]
                        ):
                        sign = u'\u2713' \
                        if sent_type_pred == sent_type_label else u'\u2718'
                        logger.info(
                            "pred = %s, label = %s %s" % (
                                id2label['sent_type'][sent_type_pred],
                                id2label['sent_type'][sent_type_label],
                                sign
                            )
                        )

                    for eval_metric in eval_metrics:
                        if result[eval_metric] > best_result[eval_metric]:
                            best_result[eval_metric] = result[eval_metric]
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                (
                                    eval_metric,
                                    str(lr), epoch,
                                    result[eval_metric] * 100.0
                                )
                            )
                            predict_for_metrics.append(eval_metric)

                    for metric_id, eval_metric in tqdm(
                        enumerate(predict_for_metrics), total=len(predict_for_metrics),
                        desc='writing predictions ... '
                    ):
                        dest_file = f'dev_best_{eval_metric}'
                        write_predictions(
                            args, eval_new_examples, eval_features, preds,
                            scores, dest_file,
                            label2id=label2id, id2label=id2label,
                            metrics=result
                        )
                        if metric_id == 0:
                            test_preds, test_result, test_scores = evaluate(
                                model, device, test_dataloader,
                                test_sent_type_labels_ids,
                                test_tags_sequence_labels_ids,
                                test_relations_sequence_labels_ids,
                                label2id, cur_train_mean_loss=None,
                                logger=None
                            )

                            output_model_file = os.path.join(
                                args.output_dir,
                                f"best_{eval_metric}_{WEIGHTS_NAME}"
                            )
                            save_model(
                                args, model, tokenizer, output_model_file
                            )
                            for metric in predict_for_metrics[1:]:
                                dest_model_path = os.path.join(
                                    args.output_dir,
                                    f"best_{metric}_{WEIGHTS_NAME}"
                                )
                                os.system(
                                    f'cp {output_model_file} {dest_model_path}'
                                )

                        dest_file = f'test_best_{eval_metric}'
                        write_predictions(
                            args, test_new_examples, test_features, test_preds,
                            test_scores, dest_file,
                            label2id=label2id, id2label=id2label,
                            metrics=test_result
                        )


            if args.log_train_metrics:
                preds, result, scores = evaluate(
                    model, device, train_dataloader,
                    sent_type_ids,
                    tags_sequence_ids,
                    relations_sequence_ids,
                    label2id, logger=logger,
                    skip_every_n_examples=args.skip_every_n_examples
                )
                result['global_step'] = global_step
                result['epoch'] = epoch
                result['learning_rate'] = lr
                result['batch_size'] = \
                    args.train_batch_size * args.gradient_accumulation_steps

                for key, value in result.items():
                    train_writer.add_scalar(key, value, global_step)

    if args.do_eval:
        test_file = os.path.join(
            args.data_dir, 'test.json'
        ) if args.test_file == '' else args.test_file
        test_examples = processor.get_test_examples(test_file)

        test_features, test_new_examples = model.convert_examples_to_features(
            test_examples, label2id, args.max_seq_length,
            tokenizer, logger, args.sequence_mode, context_mode=args.context_mode
        )
        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        test_dataloader, test_sent_type_labels_ids, \
        test_tags_sequence_labels_ids, test_relations_sequence_labels_ids = \
            get_dataloader_and_tensors(test_features, args.eval_batch_size)

        preds, result, scores = evaluate(
            model, device, test_dataloader,
            test_sent_type_labels_ids,
            test_tags_sequence_labels_ids,
            test_relations_sequence_labels_ids,
            label2id,
            compute_metrics=False
        )
        dest_file = test_file.split('/')[-1].replace('.json', '')
        write_predictions(
            args, test_new_examples, test_features,
            preds, scores, dest_file,
            label2id=label2id, id2label=id2label, metrics=result
        )


def save_model(args, model, tokenizer, output_model_file):
    start = time.time()
    model_to_save = \
        model.module if hasattr(model, 'module') else model

    output_config_file = os.path.join(
        args.output_dir, CONFIG_NAME
    )
    torch.save(
        model_to_save.state_dict(), output_model_file
    )
    model_to_save.config.to_json_file(
        output_config_file
    )
    tokenizer.save_vocabulary(args.output_dir)
    print(f'model saved in {time.time() - start} seconds to {output_model_file}')


def write_predictions(
    args, examples, features, preds,
    scores, dest_file, label2id, id2label, metrics=None
):
    aggregated_results = {}
    orig_positions_map = [ex.orig_positions_map for ex in features]
    neg_label_mapper = {
        'tags_sequence': 'O',
        'relations_sequence': '0'
    }
    for task in ['tags_sequence', 'relations_sequence']:
        aggregated_results[task] = [
            list(pred[orig_positions]) + \
            [
                label2id[task][neg_label_mapper[task]]
            ] * (len(ex.tokens) - len(orig_positions))
            for pred, orig_positions, ex in zip(
                preds[task],
                orig_positions_map,
                examples
            )
        ]

        aggregated_results[f'{task}_scores'] = [
            list(score[orig_positions]) + \
            [0.999] * (len(ex.tokens) - len(orig_positions))
            for score, orig_positions, ex in zip(
                scores[task],
                orig_positions_map,
                examples
            )
        ]

    prediction_results = {
        'idx': [
            ex.guid for ex in examples
        ],
        'tokens': [
             ' '.join(ex.tokens) for ex in examples
        ],
        'sent_type_label': [
            ex.sent_type for ex in examples
        ],
        'sent_type_pred': [
            id2label['sent_type'][x] for x in preds['sent_type']
        ],
        'sent_type_scores': [
            str(score) for score in scores['sent_type']
        ],
        'sent_start': [
            ex.sent_start for ex in examples
        ],
        'sent_end': [
            ex.sent_end for ex in examples
        ],
        'tags_sequence_labels': [
            ' '.join(ex.tags_sequence) for ex in examples
        ],
        'tags_sequence_pred': [
            ' '.join([id2label['tags_sequence'][x] if x != 0 else 'O' for x in sent])
            for sent in aggregated_results['tags_sequence']
        ],
        'tags_sequence_scores': [
            ' '.join([str(score) for score in sent])
            for sent in aggregated_results['tags_sequence_scores']
        ],
        'tags_ids': [
            ' '.join(ex.tags_ids) for ex in examples
        ],
        'relations_sequence_labels': [
            ' '.join(ex.relations_sequence) for ex in examples
        ],
        'relations_sequence_pred': [
            ' '.join([id2label['relations_sequence'][x] if x != 0 else '0' for x in sent])
            for sent in aggregated_results['relations_sequence']
        ],
        'relations_sequence_scores': [
            ' '.join([str(score) for score in sent])
            for sent in aggregated_results['relations_sequence_scores']
        ],
        'subj_start': [
            ex.subj_start for ex in examples
        ],
        'subj_end': [
            ex.subj_end for ex in examples
        ],
        'infile_offsets': [
            ' '.join([
                str(offset) for offset in ex.infile_offsets
            ]) for ex in examples
        ],
        'start_char': [
            ' '.join(ex.start_char) for ex in examples
        ],
        'end_char': [
            ' '.join(ex.end_char) for ex in examples
        ],
        'source': [
            ex.source for ex in examples
        ]
    }

    prediction_results = pd.DataFrame(prediction_results)

    prediction_results.to_csv(
        os.path.join(
            args.output_dir,
            f"{dest_file}.tsv"),
        sep='\t', index=False
    )

    if metrics is not None:
        with open(
            os.path.join(
                args.output_dir,
                f"{dest_file}_eval_results.txt"
            ), "w"
        ) as f:
            for key in sorted(metrics.keys()):
                f.write("%s = %s\n" % (key, str(metrics[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default='', type=str, required=False)
    parser.add_argument("--model", default='bert-large-uncased', type=str, required=True)
    parser.add_argument("--data_dir", default='data', type=str, required=True,
                        help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--eval_per_epoch", default=4, type=int,
                        help="How many times to do validation on dev set per epoch")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization.\n"
                             "Sequences longer than this will be truncated, and sequences shorter\n"
                             "than this will be padded.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_validate", action='store_true', help="Whether to run validation on dev set.")

    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument(
        "--eval_metrics",
        default="+".join([
            "sent_type_1_f1-score",
            "tags_sequence_macro-avg_f1-score",
            "relations_sequence_macro-avg_f1-score"
        ]),
        type=str
    )
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=6.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup.\n"
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="maximal gradient norm")

    parser.add_argument("--sent_type_clf_weight", default=1.0, type=float,
                        help="the weight of task 1")
    parser.add_argument("--tags_sequence_clf_weight", default=1.0, type=float,
                        help="The weight of task 2")
    parser.add_argument("--relations_sequence_clf_weight", default=1.0, type=float,
                        help="The weight of task 3")

    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="weight_decay coefficient for regularization")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--filter_task_3", action="store_true",
                        help="exclude task 3 from training")
    parser.add_argument("--filter_task_1", action="store_true",
                        help="exclude task 1 from training")

    parser.add_argument("--subtokens_pooling_type", type=str, default="first",
                        help="pooling mode in bert-ner, one of avg or first")
    parser.add_argument("--sequence_mode", type=str, default="not-all",
                        help="train to predict for all subtokens or not"
                        "all or not-all")
    parser.add_argument("--context_mode", type=str, default="full",
                        help="context for task 1: one from center, full, left, right")
    parser.add_argument("--lr_schedule", type=str, default="linear_warmup",
                        help="lr adjustment schedule")
    parser.add_argument("--log_train_metrics", action="store_true",
                        help="compute metrics for train set too")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="threshold for best models to save")
    parser.add_argument("--output_dirs_to_exclude", type=str, default='',
                        help="path to json file containing list of output" + \
                        " dirs to exclude fome trainig")
    parser.add_argument("--skip_every_n_examples", type=int, default=30,
                        help="number examples in train set to skip in evaluating metrics")

    parsed_args = parser.parse_args()
    main(parsed_args)
