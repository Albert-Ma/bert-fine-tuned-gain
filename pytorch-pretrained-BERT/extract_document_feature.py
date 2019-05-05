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

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

import argparse
import collections
import logging
import json
import re
import numpy as np
from tqdm import tqdm, trange

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

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, doc_orig_start, doc_stride, sentence, tokens, orig_to_tok_map, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.doc_orig_start = doc_orig_start
        self.doc_stride = doc_stride
        self.orig_to_tok_map = orig_to_tok_map
        self.sentence = sentence
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_example_to_features(example, doc_stride, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""
    # print("Length: {}" .format(len(example.text_a.split())))
    # print("Sentence: {}".format(example.text_a))
    features = []
    # The -2 accounts for [CLS], [SEP]
    max_tokens_for_doc = seq_length-2
    tok_to_orig_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.text_a.split()):
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)
    # print(all_doc_tokens)
    # print(tok_to_orig_index)
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length", "stride"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            span_bound = tok_to_orig_index[start_offset+max_tokens_for_doc]
            # find the whole token index that in this span
            for i in range(max_tokens_for_doc, 0, -1):
                if tok_to_orig_index[start_offset+i] == span_bound:
                    continue
                else:
                    length = i+1
                    break

        # find the whole token index that in this doc stride
        current_stride = doc_stride if length > doc_stride else length-start_offset-1
        # print(current_stride)
        stride_bound = tok_to_orig_index[start_offset + current_stride]
        for i in range(current_stride, 0, -1):
            if tok_to_orig_index[start_offset + i] == stride_bound:
                continue
            else:
                current_stride = i+1

        if start_offset == 0:
            doc_spans.append(_DocSpan(start=start_offset, length=length, stride=0))
        else:
            doc_spans.append(_DocSpan(start=start_offset, length=length, stride=current_stride))
        # print("start:{}, length:{}".format(start_offset, length))
        # print("doc span: {}".format(all_doc_tokens[start_offset:start_offset+length]))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += length-current_stride

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        orig_to_tok_map = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            if i >= doc_span.stride:
                if tok_to_orig_index[split_token_index] != tok_to_orig_index[split_token_index-1]:
                    orig_to_tok_map.append(len(tokens))
            tokens.append(all_doc_tokens[split_token_index])
            input_type_ids.append(0)
        orig_to_tok_map.append(len(tokens))
        tokens.append("[SEP]")
        input_type_ids.append(0)
        # print("doc span tokens: {}".format(tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        doc_orig_start = 0 if doc_span_index == 0 else tok_to_orig_index[doc_span.start+doc_span.stride]
        # logger.info("*** Example ***")
        # logger.info("unique_id: %s" % (example.unique_id))
        # logger.info("sentence: %s" % (example.text_a))
        # logger.info("doc_orig_start: %s" % (doc_orig_start))
        # logger.info("doc_stride: %s" % (doc_span.stride))
        # logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
        # logger.info("orig_to_tok_map: %s" % " ".join([str(x) for x in orig_to_tok_map]))
        # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        # logger.info(
        #     "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                sentence=example.text_a,
                doc_orig_start=doc_orig_start,
                doc_stride=doc_span.stride,
                tokens=tokens,
                orig_to_tok_map=orig_to_tok_map,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))

    return features


def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    max_length = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            if len(text_a) > max_length:
                max_length = len(text_a.split())
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    logger.info("max sentence length for {} is {}".format(input_file, max_length))
    return examples


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=64, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
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
    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    examples = read_examples(args.input_file)
    with h5py.File(args.output_file, 'w') as fout:
        example_id = 0
        sentence_to_index = {}
        for example in tqdm(examples):
            features = convert_example_to_features(
                example=example, doc_stride=args.doc_stride, seq_length=args.max_seq_length, tokenizer=tokenizer)

            unique_id_to_feature = {}
            for feature in features:
                unique_id_to_feature[feature.unique_id] = feature

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
            if args.local_rank == -1:
                eval_sampler = SequentialSampler(eval_data)
            else:
                eval_sampler = DistributedSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=len(features))

            model.eval()
            embeddings = np.zeros((len(layer_indexes), len(example.text_a.split()), 768), dtype=float)
            sentence_to_index[example.text_a] = str(example_id)
            for step, batch in enumerate(eval_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, example_indices = batch

                all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
                all_encoder_layers = all_encoder_layers

                for b, example_index in enumerate(example_indices):
                    feature = features[example_index.item()]
                    orig_to_tok_map = feature.orig_to_tok_map
                    doc_orig_start = feature.doc_orig_start
                    # print("tokens:{}, sentence:{}".format(feature.tokens, feature.sentence))
                    # SUM sub_token embedding to word embedding(word_emb)
                    for i in range(len(orig_to_tok_map)-1):
                        # print("word:{}".format(feature.sentence.split()[doc_orig_start+i]))
                        # print("token index:{}".format(i + doc_orig_start))
                        for j in range(orig_to_tok_map[i], orig_to_tok_map[i+1]):
                            # print("sub token: {}".format(feature.tokens[j]))
                            for (k, layer_index) in enumerate(layer_indexes):
                                layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
                                layer_output = layer_output[b]
                                values = [round(x.item(), 6) for x in layer_output[j]]
                                embeddings[k, i+doc_orig_start] += np.array(values)
            if len(layer_indexes) == 1:
                out = embeddings[-1]
            else:
                out = embeddings
            # print("out shape:{}".format(out.shape))
            fout.create_dataset(
                str(example_id),
                out.shape, dtype='float32',
                data=out)
            example_id += 1

        sentence_index_dataset = fout.create_dataset(
            "sentence_to_index",
            (1,),
            dtype=h5py.special_dtype(vlen=str))
        sentence_index_dataset[0] = json.dumps(sentence_to_index)


if __name__ == "__main__":
    main()
