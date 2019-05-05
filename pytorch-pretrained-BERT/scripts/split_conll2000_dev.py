import os

if __name__ == "__main__":
    data_dir = '/home/fanyixing/users/mxy/bert_rc_rep/contextual-repr-analysis/data/chunk/'
    with open(os.path.join(data_dir, 'test.txt'), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        test_sentence_num = 0
        current_sentence = []
        for i, line in enumerate(lines):
            if line.strip().startswith("-DOCSTART-"):
                continue
            elif len(line.strip()) != 0:
                current_sentence.append(line)
            elif len(line.strip()) == 0 and len(current_sentence) > 0:
                test_sentence_num += 1
                current_sentence = []
                continue
            if i == len(lines)-1 and len(current_sentence) > 0:
                test_sentence_num += 1
                current_sentence = []
    print("{} num sentences in test.txt".format(test_sentence_num))
    with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as f:
        sentences = []
        current_sentence = []
        lines = f.readlines()
        train_sentence_num = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("-DOCSTART-"):
                continue
            elif len(line.strip()) != 0:
                current_sentence.append(line)
            elif len(line.strip()) == 0 and len(current_sentence) > 0:
                train_sentence_num += 1
                sentences.append(current_sentence)
                current_sentence = []
                continue
            if i == len(lines)-1 and len(current_sentence) > 0:
                train_sentence_num += 1
                sentences.append(current_sentence)
                current_sentence = []
    print("{} num sentences in train.txt".format(train_sentence_num))

    with open(os.path.join(data_dir, 'train.txt'), 'w') as output_file:
        for sentence in sentences[:(train_sentence_num-test_sentence_num)]:
            for token_line in sentence:
                output_file.write("{}".format(token_line))
            output_file.write("\n")
    with open(os.path.join(data_dir, 'dev.txt'), 'w') as f:
        for sentence in sentences[-test_sentence_num:]:
            for token_line in sentence:
                f.write("{}".format(token_line))
            f.write("\n")
