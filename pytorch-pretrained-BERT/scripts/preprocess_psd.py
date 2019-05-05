import json
import argparse
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer


def preprocess(sentence, tokenizer, max_seq_length):
    sent = [line['word'] for line in sentence['toks']]
    # tokenize word to sub token and store ori_to_token_map
    orig_to_tok_map = []
    tokens = []
    for i, word in enumerate(sent):
        token = tokenizer.tokenize(word)
        orig_to_tok_map.append(len(tokens))
        tokens.extend(token)
    orig_to_tok_map.append(len(tokens))
    # print(tokens[:max_seq_length])
    if len(tokens) > max_seq_length:
        pre_word_index = 0
        for i, index in enumerate(orig_to_tok_map):
            if index < max_seq_length:
                pre_word_index = i
            else:
                break
        sentence['toks'] = sentence['toks'][:pre_word_index]
        # print(pre_word_index)
        swes_dict = {}
        for index, value in sentence['swes'].items():
            if int(index) <= pre_word_index:
                swes_dict[index] = value
        sentence['swes'] = swes_dict
    return sentence


def main(tokenizer):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        sentences = []
        # lines = f.readlines()
        # print(len(data))
        for i, sents in enumerate(tqdm(data)):
            sentences.append(preprocess(sents, tokenizer, args.max_seq_length))

    with open(args.output_file, 'w') as output_file:
        json.dump(sentences, output_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")


    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False)
    main(tokenizer)
