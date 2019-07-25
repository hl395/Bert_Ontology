import conlleval
import codecs

output_predict_file = "OUTPUT/ner/test_results.tsv"
output_predict_file_processed = "OUTPUT/ner/test_results_processed.tsv"

with codecs.open(output_predict_file, 'r', encoding='utf-8') as f:
    counter = 0
    line_1 = []
    line_2 = []
    line_3 = []
    lines = ''
    for line in f:
        if line.strip():
            content = line.strip()
            tokens = content.split('\t')
            if counter == 0:
                line_1 = tokens
                counter += 1
            elif counter == 1:
                line_2 = tokens
                counter += 1
            elif counter == 2:
                line_3 = tokens
        else:
            for a, b, c in zip(line_1, line_2, line_3):
                if a not in ["[PAD]", "[CLS]", "[SEP]"] and b not in ["X", "APAD"]:
                    lines += a + " " + b + " " + c + '\n'
            counter = 0



with codecs.open(output_predict_file_processed, 'w', encoding='utf-8') as writer:
    writer.write(lines + '\n')



eval_result = conlleval.return_report(output_predict_file_processed)
print(''.join(eval_result))