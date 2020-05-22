import tensorflow as tf
import csv
import numpy as np

def read_eval_result(filename):
    result = []
    with open(filename, "r", encoding='utf-8') as record_file:
        line = record_file.readline()
        records = line.split('\t')
        if len(records)==2:
            if records[0] > records[1]:
                result.append(0)
            else:
                result.append(1)

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
    return lines

import os.path

lines = read_tsv(os.path.dirname(__file__) +"/../glue_data/MYTESTING/Iteration_10/clinical_finding_hier_test.tsv")  # clinical finding

lines = lines[1:]
true_result = []
for line in lines:
    line = line[-1]
    true_result.append(line)

# print(true_result)

predict_result = []

lines = read_tsv('clinical_finding/simple_pre_training_100000/test_results.tsv')

for line in lines:
    if line:
        line = [float(x) for x in line]
        max_index = line.index(max(line))
        predict_result.append(str(max_index))


print("classification report is: ")
from sklearn.metrics import classification_report
print(classification_report(true_result, predict_result, labels=[0, 1], digits=3))



predicted_prob = []
for line in lines:
    if line:
        line = [float(x) for x in line]
        predicted_prob.append(line)

# print(predicted_prob)
predicted_prob = np.array(predicted_prob)

predicted_prob = predicted_prob[:, 1]
# print(predicted_prob)
# Now we're going to assess the quality of the neural net using ROC curve and AUC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

true_result = [int(x) for x in true_result]
# send the actual dependent variable classifications for param 1,
# and the confidences of the true classification for param 2.
FPR, TPR, _ = roc_curve(true_result, predicted_prob)

# Calculate the area under the confidence ROC curve.
# This area is equated with the probability that the classifier will rank
# a randomly selected defaulter higher than a randomly selected non-defaulter.
AUC = auc(FPR, TPR)

# What is "good" can dependm but an AUC of 0.7+ is generally regarded as good,
# and 0.8+ is generally regarded as being excellent
print("AUC is {}".format(AUC))

# Now we'll plot the confidence ROC curve
plt.figure()
plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
# plt.savefig(img_path + img_name + 'roc2.png')
plt.show()



true_result = [int(x) for x in true_result]
predict_result = [int(x) for x in predict_result]


from sklearn.metrics import fbeta_score
# print(fbeta_score(true_result, predict_result, beta=1))
print('F1 macro avg is:')
print(fbeta_score(true_result, predict_result, average='macro', beta=1))
print('F1 micro avg is:')
print(fbeta_score(true_result, predict_result, average='micro', beta=1))
# print(fbeta_score(true_result, predict_result, average='weighted', beta=1))

print("F2 scores are:")
print(fbeta_score(true_result, predict_result, beta=2))
print(fbeta_score(true_result, predict_result, average='macro', beta=2))
print(fbeta_score(true_result, predict_result, average='micro', beta=2))
print(fbeta_score(true_result, predict_result, average='weighted', beta=2))
