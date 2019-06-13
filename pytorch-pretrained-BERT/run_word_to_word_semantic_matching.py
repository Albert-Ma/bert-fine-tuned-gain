# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import re
import math
from numpy import dot
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm, trange
from scipy.stats import pearsonr, spearmanr

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, word_a, text_a, word_b, text_b, label=None):
        self.unique_id = unique_id
        self.word_a = word_a
        self.text_a = text_a
        self.word_b = word_b
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, word_a_orig_to_tok_map,
                 word_b_orig_to_tok_map, input_ids, input_mask,
                 input_type_ids, label_id):
        self.unique_id = unique_id
        self.tokens = tokens
        self.word_a_orig_to_tok_map = word_a_orig_to_tok_map
        self.word_b_orig_to_tok_map = word_b_orig_to_tok_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.label_id = label_id


def convert_examples_to_features(examples, seq_length, tokenizer, use_sentence_b=True):
    """Loads a data file into a list of `InputFeature`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        # TODO: delete useless sentence which word a or b not in it.
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b and use_sentence_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b and use_sentence_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"

            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                # tokens_a = tokens_a[0:(seq_length - 2)]
                raise ValueError("sentence length {} is greater than max_seq_length {}"
                                 .format(len(tokens_a), seq_length))

        truncated = False
        subtokens = []
        subtokens.append("[CLS]")
        word_a = example.word_a
        word_a_orig_token_map = []
        for (i, orig_token) in enumerate(example.text_a.split()):
            # TODO: now we only use the first one
            if orig_token == word_a and len(word_a_orig_token_map) == 0:
                word_a_orig_token_map.append(len(subtokens))
                subtokens.extend(tokenizer.tokenize(orig_token))
                word_a_orig_token_map.append(len(subtokens))
                # check if this word has be truncated
                if len(subtokens) > len(tokens_a):
                    truncated = True
            else:
                subtokens.extend(tokenizer.tokenize(orig_token))
        if len(word_a_orig_token_map) == 0 or truncated:
            continue
        subtokens.append("[SEP]")

        word_b = example.word_b
        word_b_orig_token_map = []
        for (i, orig_token) in enumerate(example.text_b.split()):
            if orig_token == word_b:
                word_b_orig_token_map.append(len(subtokens))
                subtokens.extend(tokenizer.tokenize(orig_token))
                word_b_orig_token_map.append(len(subtokens))
                # check if this word has be truncated
                if len(subtokens) > len(tokens_a+tokens_b):
                    truncated = True
                break
            else:
                subtokens.extend(tokenizer.tokenize(orig_token))
        if len(word_a_orig_token_map) == 0 or truncated:
            continue
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0]
        input_mask += [1] * (len(input_ids) - 1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        label_id = float(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("sentence: %s" % " ".join(str(x) for x in tokens))
            logger.info("word_a: %s" % (example.word_a))
            logger.info("word_a_orig_token_map:%s" % " ".join([str(x) for x in word_a_orig_token_map]))
            logger.info("word_b: %s" % (example.word_b))
            logger.info("word_b_orig_token_map:%s" % " ".join([str(x) for x in word_b_orig_token_map]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
            logger.info("label: %s" % (label_id))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                word_a_orig_to_tok_map=word_a_orig_token_map,
                word_b_orig_to_tok_map=word_b_orig_token_map,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file, use_sentence_b=False, task='base'):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    # max_length = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            # print(line)
            label = 1
            if task == 'scws':
                word_a, word_b, sentence_a, sentence_b, label = line.split('\t')
                # We only do one single verb word.
                if len(word_a.strip().split()) != 1 or \
                        len(word_b.strip().split()) != 1:
                    continue
            else:
                line = line.split('\t')
                if len(line) != 4:
                    continue
                word_a = line[0]
                word_b = line[1]
                sentence_a = line[2]
                sentence_b = line[3]
                # We only do one single verb word.
                if len(word_a.strip().split()) != 1 or \
                        len(word_b.strip().split()) != 1:
                    continue
            examples.append(
                InputExample(unique_id=unique_id, word_a=word_a, text_a=sentence_a,
                             word_b=word_b, text_b=sentence_b, label=float(label)))
            unique_id += 1
    # logger.info("max sentence length for {} is {}".format(input_file, max_length))
    return examples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=False)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='base',
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--use_sentence_b", action='store_true', help="use sentence b's embedding")
    parser.add_argument("--threshold",
                        default=0.15,
                        type=float,
                        help="threshold for restricted improvements.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    layer_indexes = [int(x)+12 for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    examples = read_examples(args.input_file, args.use_sentence_b, task=args.task_name)

    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length,
        tokenizer=tokenizer, use_sentence_b=args.use_sentence_b)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_input_type_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_type_ids, all_input_mask, all_example_index)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model.eval()
    pearson_corr = []
    spearman_corr = []
    label_id = []
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Generating")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, token_type_ids, input_mask, example_indices = batch

        all_encoder_layers, _ = model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers

        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            word_a_orig_to_tok_map = feature.word_a_orig_to_tok_map
            word_b_orig_to_tok_map = feature.word_b_orig_to_tok_map
            embedding_a = np.zeros((len(layer_indexes), 768), dtype=np.float64)
            embedding_b = np.zeros((len(layer_indexes), 768), dtype=np.float64)
            for i in range(len(word_a_orig_to_tok_map)-1):
                for j in range(word_a_orig_to_tok_map[i], word_a_orig_to_tok_map[i+1]):
                    for (k, layer_index) in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layer_output = layer_output[b]
                        values = [round(x.item(), 6) for x in layer_output[j]]
                        embedding_a[k] += np.array(values)
            for i in range(len(word_b_orig_to_tok_map)-1):
                for j in range(word_b_orig_to_tok_map[i], word_b_orig_to_tok_map[i+1]):
                    for (k, layer_index) in enumerate(layer_indexes):
                        layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                        layer_output = layer_output[b]
                        values = [round(x.item(), 6) for x in layer_output[j]]
                        embedding_b[k] += np.array(values)
            pearson_score = []
            spearman_score = []
            label_id.append(feature.label_id)
            for i in range(len(embedding_a)):
                assert embedding_a[i].size == embedding_b[i].size
                cos_sim = dot(embedding_a[i], embedding_b[i]) / (norm(embedding_a[i]) * norm(embedding_b[i])+1e-6)
                pearson_score.append(cos_sim)
                spearman_score.append(cos_sim)

                # logger.info("layer {} 's {}th person score:{}, spearman score:{}"
                #             .format(i, feature.unique_id, pearson_score[i], spearman_score[i]))
            # replace nan TO 0
            pearson_score = np.nan_to_num(pearson_score, copy=False).astype(np.float64)
            spearman_score = np.nan_to_num(spearman_score, copy=False).astype(np.float64)

            pearson_corr.append(pearson_score)
            spearman_corr.append(spearman_score)
    print("spearman_corr here:{}".format(len(spearman_corr)))

    pearson_corr = np.array(pearson_corr)
    spearman_corr = np.array(spearman_corr)
    if args.task_name == 'scws':
        pearson_corr = np.split(pearson_corr, len(layer_indexes), axis=1)
        spearman_corr = np.split(spearman_corr, len(layer_indexes), axis=1)
        pearson = []
        spearman = []
        for i in range(len(layer_indexes)):
            pearson.append(pearsonr(np.reshape(pearson_corr[i], -1), label_id)[0])
            spearman.append(spearmanr(np.reshape(spearman_corr[i], -1), label_id)[0])
    else:
        pearson = np.average(pearson_corr, axis=0)
        spearman = np.average(spearman_corr, axis=0)

    logger.info("pearson_corr: %s" % " ".join([str(x) for x in pearson]))
    logger.info("spearman_corr: %s" % " ".join([str(x) for x in spearman]))


if __name__ == "__main__":
    main()
