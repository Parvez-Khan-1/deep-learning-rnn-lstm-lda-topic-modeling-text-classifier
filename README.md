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
