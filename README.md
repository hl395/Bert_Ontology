# Bert_Ontology
Experiment with training BERT for ontology engeneering.

## NJIT Kong usage
1. Go to [njit software download](http://ist.njit.edu/software-available-download/)
2. Download and install MobaXterm (for Windows user only).
3. SSH to Kong.njit.edu with port 22.
![alt text](TestBert/image/SSH Connection.PNG)

## 1. Environment configuration:
### 1.1. Hardware requirement (Recommended or better)
* NVIDIA RTX Titan with 24GB GDDR6 memory
* Driver version == 418.56
* CUDA version == 10.1

### 1.2. Software requirement
* Python 3.6 
* Keras==2.2.4
* matplotlib==3.0.2
* nltk==3.4.1
* numpy==1.16.4
* pandas==0.23.0
* scikit-learn==0.21.2
* tensorflow-gpu==1.13.1\
To install required python modules, run `pip3 install -r requirements.txt`


## 2.	Source code
The source code is maintained on GitHub repository with the link below:\
https://github.com/hl395/Bert_Ontology\
The pre-trained BERT model, released by Google can be found at:\
https://github.com/google-research/bert\


## 3.	Dataset format
### 3.1.	Pre-training data format
In our task, the pre-training data is concept triples. Each triple contains three concepts, with one concept per line. An empty line is used to separate triples. In cases a focus concept has no parent or child, there are only two concepts in the corresponding triple. 
An example of pre-training data is as follows:
```
congenital anomaly of aorta
congenital stenosis of aorta
congenital supravalvular aortic stenosis

genodermatosis
inherited cutaneous hyperpigmentation
naegeli-franceschetti-jadassohn syndrome

hyperpigmentation of skin
inherited cutaneous hyperpigmentation
terminal osseous dysplasia and pigmentary defect syndrome
```
A short version of the example file for the pre-training data can be found at /data/pre_training_data_example.txt in the GitHub repository.  
     
### 3.2.	Fine-tuning data format
For fine-tuning, we need three files: [train.tsv](TestBert/data/train.tsv) for training, [dev.tsv](TestBert/data/dev.tsv) for validation, and [test.tsv](TestBert/data/test.tsv) for prediction. The “train.tsv” and “dev.tsv” files share the same format while the “test.tsv” is different by hiding the true labels.

To fine-tune BERT as IS-A relationship classifier, we extract IS-A connected concept pairs as positive training sample, and concept pairs that are not connected as negative training sample. Each concept pair is recorded as one string in one line, with the two concepts’ ids and names, and the IS-A label of this pair. The information is organized into five columns: 
•	Column “Quality” indicates the IS-A label between the two concepts. 
•	Column “#1 ID” represents the SNOMED ID of the first concept.
•	Column “#2 ID” represents the SNOMED ID of the second concept.
•	Column “#1 String” represents the SNOMED name of the first concept.
•	Column “#2 String” represents the SNOMED name of the second concept.
Columns are separated using Tab as the delimiter. 
An example of the fine-tuning data is as follows:
```
Quality	#1 ID	#2 ID	#1 String	#2 String
1	366054000	301976001	finding of fluorescein tear drainage	fluorescein tear drainage impaired
0	295116004	295019008	allergy to chymotrypsin	allergy to mannitol
```
A short version of the three example files, “train.tsv”, “dev.tsv”, and “test.tsv” for the pre-training data can be found at the “data” directory in the GitHub repository.

### 3.3.	Test data format
To test the trained IS-A relationship classifier, we extract both IS-A connected concept pairs as positive testing sample, and concept pairs that are not connected as negative testing sample. Each concept pair is recorded as one string in one line, with the two concepts’ ids and names. The information is organized into five columns: 
•	Column “Quality” indicates the IS-A label between the two concepts. 
•	Column “#1 ID” represents the SNOMED ID of the first concept.
•	Column “#2 ID” represents the SNOMED ID of the second concept.
•	Column “#1 String” represents the SNOMED name of the first concept.
•	Column “#2 String” represents the SNOMED name of the second concept.
Columns are separated using Tab as the delimiter. 
Note that the IS-A label of this pair is also included for evaluation simplicity. The true label is not visible or used in testing our classifier.
An example of the fine-tuning data is as follows:
```
index	#1 ID	#2 ID	#1 String	#2 String
0	400186008	773629001	neoplasm of integumentary system	onychomatricoma	1
1	298756009	316561000119102	finding of bone of upper limb	osteophyte of left elbow	1
…… 
6734	51868009	735563002	duodenal ulcer disease	cicatrix of middle ear	0
6735	762366009	735906001	prolapse of left eye co-occurrent with laceration	effects of water pressure	0
……
```
A short version of the example file for the pre-training data can be found at /data/testing_data_example.csv in the GitHub repository.


## 4.	Execute program
### 4.1.	Preparation
* a)	The program repository can be cloned using command:
`git clone https://github.com/hl395/Bert_Ontology.git`
The hardware compatibility and software requirement should be verified before executing the program.  
* b)	Download the pre-trained BERT model, e.g. BERTBASE uncased model
BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters from:  https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
The downloaded BERT model should include the “vocab.txt” file and “bert_config.json” and three bert checkpoint files, “bert_model.ckpt.meta”, “bert_model.ckpt.index”, and “bert_model.ckpt.data-00000-of-00001”. 

### 4.2.	Pre-training
#### 4.2.1.	Create pre-training data
The parameters used to control this data creation are specified in the “creating_pretraining_data.py”. The required parameters include:
  FLAGS.input_file = “/path/to/pre-training_data_example.txt”
  FLAGS.output_file = “/path/to/tf_examples.tfrecord” 
  FLAGS.vocab_file = “/path/to/downloaded_BERT_model/vocab.txt”
  FLAGS.do_lower_case = True
  FLAGS.max_seq_length = 128
  FLAGS.max_predictions_per_seq = 20
  FLAGS.masked_lm_prob = 0.15
  FLAGS.random_seed = 12345
  FLAGS.dupe_factor = 5

The usage of parameters can be referred at the vanilla BERT GitHub page.
To generate pre-training data, run “python creating_pretraining_data.py”.
After the pre-training data is generated, it is wrote to the output directory named by “tf_examples.tfrecord.” 
 
#### 4.2.2.	Run pre-training 
The parameters used to control pre-training are specified in the “run_pretraining.py”. The required parameters include:
  FLAGS.input_file = “/path/to/ tf_examples.tfrecord” (from 4.2.1)
  FLAGS.output_dir = “/path/to/pre_trained_model_directory” 
  FLAGS.vocab_file = “/path/to/downloaded_BERT_model/vocab.txt” (the same as in 4.2.1)
  FLAGS.do_train = True (perform training)
  FLAGS.do_eval = True (perform evaluation/validation)
  FLAGS.bert_config_file = “/path/to/downloaded_BERT_model/bert_config.json”
  FLAGS.init_checkpoint = “/path/to/downloaded_BERT_model/bert_model.ckpt”
  FLAGS.train_batch_size = 64   
  FLAGS.max_seq_length = 128
  FLAGS.max_predictions_per_seq = 20
  # FLAGS.num_train_steps = 15000
  FLAGS.num_train_steps = 100000
  FLAGS.num_warmup_steps = 5000
  FLAGS.save_checkpoints_steps = 20000
  FLAGS.learning_rate = 2e-5

The usage of parameters can be referred at the vanilla BERT GitHub page.
To pre-training the downloaded BERT with our own corpus, run “python run_pretraining.py”.
After pre-training the BERT model, the obtained new model is saved to the output directory in “/path/to/pre_trained_model_directory” where the value of “FLAGS.output_dir.” Note that the obtained new model’s name could vary depends on the number of training steps are used. However, the model still consists of three files and checkpoint file. An example using the parameters above will generate a model with three files as follows: “model.ckpt-100000.meta”, “model.ckpt-100000.index”, and “model.ckpt-100000.data-00000-of-00001” with the same number 100000 in their names as it is the value used as the number of training steps.

### 4.3.	Fine-tuning
To fine-tune the obtained model from 4.2, we run “run_classifier_hao.py” with specifying the following required parameters:
FLAGS.bert_config_file = “/path/to/downloaded_BERT_model/bert_config.json”
FLAGS.vocab_file = “/path/to/downloaded_BERT_model/vocab.txt"
FLAGS.init_checkpoint = “/path/to/pre_trained_model_directory” (in 4.2) 
FLAGS.data_dir = “/path/to/fine_tuning_data_directory/” (the directory that contains the both fine-tuning training, evaluation, and testing data)
FLAGS.output_dir = “/path/to/fine_tuned_model_directory/"
FLAGS.train_batch_size = 64  
FLAGS.max_seq_length = 128
FLAGS.num_train_epochs = 3
FLAGS.do_train = True (perform training)
FLAGS.do_eval = True (perform evaluation/validation)
FLAGS.do_predict = True (perform prediction)
FLAGS.task_name = "MRPC" 

The usage of parameters can be referred at the vanilla BERT GitHub page.
To fine-tune the pre-trained BERT with concept pairs, run “python run_classifier_hao.py”.
After fine-tuning the BERT model as an IS-A relationship classifier, the obtained classifier is saved to the output directory in “/path/to/fine_tuned_model_directory” where the value of “FLAGS.output_dir” is specified. 

### 4.4.	Testing
The testing is performed after the model is fine-tuned in 4.3, because we turn on the flag for prediction by setting “FLAGS.do_predict = True.” The fine-tuned model’s prediction is saved in the “test_results.tsv” file in the directory of “/path/to/fine_tuned_model_directory.” Note that we only need to fine-tune the model once, and then use the obtained model to test on different testing data. This can be achieved by update the testing sample, i.e. “test.tsv” file with new testing data, and set the “FLAGS.do_train = False” and “FLAGS.do_eval = False.”

### 4.5.	Evaluation
For each testing concept pair, our model predicts the probabilities that the two concepts should be connected by IS-A and not, respectively. The results are recorded in the “test_results.tsv” file. Use “output/read_results.py” to read the true results and prediction results to evaluate the model’s performance. The metrics used including Precision, Recall, F1 and F2 scores, the micro average and macro average of these metrics are also calculated. 





