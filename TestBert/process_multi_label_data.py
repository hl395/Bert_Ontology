import smart_open
import tensorflow as tf
import csv
import numpy as np

# hier = "clinical_finding"
hier = "procedure"


input_file = "ontHierarchy_sno_" + hier + "_MultiLabel_bert.txt"
new_test_input_file = "ontHierarchy_sno_" + hier + "_MultiLabel_new_concepts.txt"

# with smart_open.smart_open(input_file) as f:
#     for i, line in enumerate(f):
#         # get the id for each concept paragraph
#         splitted = line.decode("iso-8859-1").split("\t")


def read_file(input_file):
    with open(input_file, "r", encoding='iso-8859-1') as f:
        reader = csv.reader(f, delimiter="\t")
        lines = []
        for line in reader:
            lines.append(line)
        return lines

lines = read_file(input_file)


from collections import namedtuple
Concepts_pair = namedtuple("Concepts_pair", ["source", "target"])

def read_into_dict(lines):
    concepts_pair_dict = {}
    for (i, line) in enumerate(lines):
        line = list(filter(None, line))
        if i == 0:
            relationship_list = line[::2]
            print(relationship_list)
            rel_types = len(relationship_list)
        cp = Concepts_pair(line[1], line[5])
        if cp in concepts_pair_dict:
            index = relationship_list.index(line[2])
            concepts_pair_dict[cp][index] = 1
        else:
            labels = np.zeros((rel_types,), dtype=int)
            index = relationship_list.index(line[2])
            labels[index] = 1
            concepts_pair_dict[cp] = labels
    return concepts_pair_dict

concepts_pair_dict = read_into_dict(lines)

def process_dict(concepts_pair_dict):
    index = 0
    count = 0
    result_lines = []
    for key, value in concepts_pair_dict.items():
        index += 1
        result_line = "{}\t{}\t{}\t{}\n".format(index, key.source, key.target, value)
        sum = 0
        for v in value:
            sum += v
        if sum > 2:
            count += 1
            print("key: {}, value: {}".format(key, value))
        result_lines.append(result_line)
    print(len(lines))
    print(len(result_lines))
    return result_lines

result_lines = process_dict(concepts_pair_dict)

def write_test_tsv(filename, records):
    with open(filename, "w", encoding='utf-8') as record_file:
        record_file.write("index\t#1 String\t#2 String\t#Label List\n")
        for record in records:
            record_file.write(str(record))


from sklearn.model_selection import train_test_split
train_examples, test_examples = train_test_split(result_lines, test_size=0.05, shuffle=True)

train_examples, validation_examples = train_test_split(train_examples, test_size=0.1, shuffle=True)

filename = hier + '_train.tsv'
write_test_tsv(filename, train_examples)

filename = hier + '_dev.tsv'
write_test_tsv(filename, validation_examples)

filename = hier + '_test.tsv'
write_test_tsv(filename, test_examples)




new_test_lines = read_file(new_test_input_file)
new_test_concepts_pair_dict = read_into_dict(new_test_lines)
new_test_result_lines = process_dict(new_test_concepts_pair_dict)

filename = hier + '_new_test.tsv'
write_test_tsv(filename, new_test_result_lines)
