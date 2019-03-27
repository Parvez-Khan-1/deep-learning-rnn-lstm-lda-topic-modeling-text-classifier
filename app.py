# !pip install keras-metrics

import os
import json
import argparse
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import keras_metrics
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def file_to_data(filename, content='text', label='label', n_most_common_words=8000):
  data = pd.read_csv(filename, usecols=[content, label])
  print(data[label].value_counts())
  if min(data[label]) > 0:
    data[label] = data[label] - 1
  y = data[label].values
  X = data[content].values
  for i,val in enumerate(X):
    if type(val) != str:
      X = np.delete(X, i)
      y = np.delete(y, i)
  y = to_categorical(y)
  n_most_common_words = 8000
  max_len = 130
  tokenizer = Tokenizer(num_words=n_most_common_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
  tokenizer.fit_on_texts(X)
  sequences = tokenizer.texts_to_sequences(X)
  word_index = tokenizer.word_index
  X = pad_sequences(sequences, maxlen=max_len)
  X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.25, random_state=42)
  return X_train, X_test, y_train, y_test

def plot_cm(cm, num_classes):
  classes = [str(i) for i in range(num_classes)]
  normalize = False
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title('Confusion Matrix')
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()
  plt.show()
  
def train(filenames, epochs, num_classes, emb_dim, batch_size, n_most_common_words):
  earlystop = EarlyStopping(monitor='val_loss',patience=20, min_delta=0.0001)
  output = []
  for filename in filenames:
    checkpointer = ModelCheckpoint(
          filepath='models/{}_{}.hfs5'.format(filename.split('/')[-1],epochs),
          save_best_only=True)
    X_train, X_test, y_train, y_test = file_to_data(filename, 'text', 'label', n_most_common_words)
    model = Sequential()
    model.add(Embedding(n_most_common_words, emb_dim, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2,callbacks=[checkpointer])
    accr = model.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    output.append([model, history, accr])
  return output
  
def get_metrics(y_true, y_pred):
    label = [0,1,2,3,4]
    return precision_recall_fscore_support(y_true,y_pred,labels=label,average="micro")

parser = argparse.ArgumentParser(description='This script trains or evaluate a model.')
parser.add_argument('--train', default=None, help="Filepath of the train file with the 'text' and 'label' headers.")
parser.add_argument('--test', default=None, help="Filepath of the test file with the 'text' and 'label' headers.")
parser.add_argument('--model', default=None, help="Filepath of the hfs5 file.")
parser.add_argument('--epochs', default=100, type=int, help="Number of epoches to run.")
parser.add_argument('--num_classes', default=2, type=int, help="Number of classes.")
parser.add_argument('--num_words', default=8000, type=int, help="Number of common words.")
parser.add_argument('--emb_dim', default=128, type=int, help="Number of embedded dimension.")
parser.add_argument('--batch_size', default=256, type=int, help="Size of the batch.")
args = parser.parse_args()

num_classes = args.num_classes
if args.train is not None:
  filenames = args.train.split(',')
  epochs = args.epochs
  n_most_common_words = args.num_words
  emb_dim = args.emb_dim
  batch_size = args.batch_size
  models = train(filenames, epochs, num_classes, emb_dim, batch_size, n_most_common_words)
elif args.model is not None and args.test is not None:
  models = args.model.split(',')
  filenames = args.test.split(',')
  for model_file in models:
    model = load_model(model_file)
    for filename in filenames:
      print(model_file, filename)
      X_train, X_test, y_train, y_test = file_to_data(filename)
      y_pred = model.predict(X_test)
      y_pred_classes = np.argmax(y_pred, axis=1)
      y_test_classes = np.argmax(y_test, axis=1)
      cm = confusion_matrix(y_test_classes, y_pred_classes)
      print('Precision, recall, fscore, support:', get_metrics(y_test_classes, y_pred_classes))
      accr = model.evaluate(X_test,y_test)
      print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
      plot_cm(cm, num_classes)
      json.dump(accr, open('results/%s_%s.json'%(model_file.split('/')[-1],filename.split('/')[-1]),'w'))