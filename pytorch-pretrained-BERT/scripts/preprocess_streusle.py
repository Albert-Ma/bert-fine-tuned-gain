import argparse
from tqdm import tqdm

from pytorch_pretrained_bert.tokenization import BertTokenizer


def preprocess(sentence, tokenizer, max_seq_length):
    sent = [line.split()[0] for line in sentence]
    # tokenize word to sub token and store ori_to_token_map
    orig_to_tok_map = []
    tokens = []
    for i, word in enumerate(sent):
        token = tokenizer.tokenize(word)
        orig_to_tok_map.append(len(tokens))
        tokens.extend(token)
    orig_to_tok_map.append(len(tokens))
    if len(tokens) > max_seq_length:
        pre_word_index = 0
        for i, index in enumerate(orig_to_tok_map):
            if index < max_seq_length:
                pre_word_index = i
            else:
                break
        return sentence[:pre_word_index]
    else:
        return sentence


def main(tokenizer):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        sentences = []
        current_sentence = []
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines)):
            if line.strip().startswith("-DOCSTART-"):
                continue
            elif len(line.strip()) != 0:
                current_sentence.append(line)
            elif len(line.strip()) == 0 and len(current_sentence) > 0:
                sentences.append(preprocess(current_sentence, tokenizer, args.max_seq_length))
                current_sentence = []
                continue
            if i == len(lines)-1 and len(current_sentence) > 0:
                sentences.append(preprocess(current_sentence, tokenizer, args.max_seq_length))
                current_sentence = []
    with open(args.output_file, 'w') as output_file:
        for sentence in sentences:
            for token_line in sentence:
                output_file.write("{}".format(token_line))
            output_file.write("\n")


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
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    main(tokenizer)
