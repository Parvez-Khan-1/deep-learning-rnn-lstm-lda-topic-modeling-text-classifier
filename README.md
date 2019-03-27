# deep-learning-rnn-lstm-lda-topic-modeling-text-classifier
Deep learning and Topic Modeling approaches mixed for text classification

## Instructions

For training you need to prepare a CSV file with two columns, with the 'text' and 'label' headers, and run it from the command line as shown below:

```sh
python app.py --train data/train_all.csv
```

For testing a model you need to indicate the hfs5 model path and a CSV file with the 'text' and 'label' headers, as shown below:


```sh
python app.py --model models/YOUR_MODEL.hfs5 --test data/test_all.csv
```

The supported command line parameters are the following:

```sh
usage: app.py [-h] [--train TRAIN] [--test TEST] [--model MODEL]
              [--epochs EPOCHS] [--num_classes NUM_CLASSES]
              [--num_words NUM_WORDS] [--emb_dim EMB_DIM]
              [--batch_size BATCH_SIZE]

This script trains or evaluate a model.

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Filepath of the train file with the 'text' and 'label'
                        headers.
  --test TEST           Filepath of the test file with the 'text' and 'label'
                        headers.
  --model MODEL         Filepath of the hfs5 file.
  --epochs EPOCHS       Number of epoches to run.
  --num_classes NUM_CLASSES
                        Number of classes.
  --num_words NUM_WORDS
                        Number of common words.
  --emb_dim EMB_DIM     Number of embedded dimension.
  --batch_size BATCH_SIZE
                        Size of the batch.
```
