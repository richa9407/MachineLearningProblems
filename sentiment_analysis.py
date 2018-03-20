
# coding: utf-8
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pandas as pd
import re


def preprocess_text(raw_string):
  #Eliminate conflicting alphanum words
  raw_string = re.sub(r'\.( [a-z])', '\g<1>', raw_string) #A period followed by a lowercase isnt a new sentence
  raw_string = re.sub(r'\'s','', raw_string) #Remove "'s" from for eg Apple's

  #Keep only Alpha
  raw_string = re.sub(r'[!,\(\)\[\]"\'—?:;]','', raw_string)
  raw_string = re.sub(r'[-/]', ' ', raw_string)

  raw_string = raw_string.lower()
  #Handle additional spaces
  raw_string = re.sub(r' {2,}', ' ', raw_string)
  
  #Handle additional periods
  raw_string = re.sub(r'\.{1,}',' ',raw_string)
  return raw_string

with open('Data_sets/textTrainData.txt') as f:
  sentences = f.read()
  sentences = sentences.split('\n')

proc_sent = list() 
for sentence in sentences:
  temp = sentence.split('\t')
  try:
    proc_sent.append([preprocess_text(temp[0]), int(temp[1])])
  except:
    pass

test_portion = int(0.7*len(proc_sent))
test = proc_sent[:test_portion]
sentences = proc_sent

# Generate counts from text using a vectorizer.  There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.
vectorizer = CountVectorizer(stop_words='english')
train_features = vectorizer.fit_transform([s[0] for s in sentences])
test_features = vectorizer.transform([s[0] for s in test])

# Fit a naive bayes model to the training data.
# This will train the model using the word counts we computer, and the existing classifications in the training set.
nb = MultinomialNB()
nb.fit(train_features, [int(s[1]) for s in sentences])

# Now we can use the model to predict classifications for our test features.
predictions = nb.predict(test_features)

actual = [int(t[1]) for t in test]
print('::RESULTS FOR TRAINING DATASET::')
print(pd.crosstab(pd.Series(actual),pd.Series(predictions),rownames=['True'],colnames=['Predicted'],margins=True))
print('')
print('Accuracy:',metrics.accuracy_score(actual,predictions))
print('Precision:',metrics.precision_score(actual,predictions))
print('Recall:',metrics.recall_score(actual,predictions))
print('F1 Score:',metrics.f1_score(actual,predictions))


#FOR TEST DATASET

with open('Data_sets/textTestData.txt') as f:
  test_sentences = f.read()
  test_sentences = test_sentences.split('\n')

proc_sent_test = list() 

for sentence in test_sentences:
  temp = sentence.split('\t')
  try:
    proc_sent_test.append([preprocess_text(temp[0]), int(temp[1])])
  except:
    pass

test_sentences = proc_sent_test

# Generate counts from text using a vectorizer.  There are other vectorizers available, and lots of options you can set.
# This performs our step of computing word counts.
features = vectorizer.transform([s[0] for s in test_sentences])

predictions = nb.predict(features)
actual = [int(t[1]) for t in test_sentences]
print('\n::RESULTS FOR TEST DATASET::')
print(pd.crosstab(pd.Series(actual),pd.Series(predictions),rownames=['True'],colnames=['Predicted'],margins=True))
print('')
print('Accuracy:',metrics.accuracy_score(actual,predictions))
print('Precision:',metrics.precision_score(actual,predictions))
print('Recall:',metrics.recall_score(actual,predictions))
print('F1 Score:',metrics.f1_score(actual,predictions))
