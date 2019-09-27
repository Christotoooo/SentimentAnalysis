#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Christopher Zheng
#260760794

import random
from sklearn.metrics import confusion_matrix
import numpy as np
import nltk
from nltk.corpus import stopwords as sw
from nltk.stem import *
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from scipy.sparse import csc_matrix, dia_matrix, lil_matrix
import re

stopwords = set(sw.words("english"))
#print(stopwords)
suggestive_words=["aren't","didn't","against","couldn","isn't","wouldn",                  "doesn","weren","weren't","ain","should","hasn","mustn","haven't",                   "hadn't", "doesn't",'didn',"shouldn't",'isn','needn',"won't",'no',                  "wasn't", 'don',"hasn't",'hadn',"mightn't",'mightn','not',"don't",                   "couldn't",'haven',"mustn't",'nor','aren','wasn', "wouldn't", "should've"]
stopwords = stopwords - set(suggestive_words)


def generate_unigrams(s):
    # Convert to lowercases. Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s.lower())
    # Break sentence in the token, remove empty tokens and stopwords
    tokens = [token for token in s.split(" ") if token != "" and token not in stopwords]

    #lemmatize
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    
#      # stem
#     tokens = [PorterStemmer().stem(token) for token in tokens]
    
    # Concatentate the tokens into unigrams and return
    unigrams = zip(*[tokens[i:] for i in range(1)])
    return [" ".join(unigram) for unigram in unigrams]

#print(generate_unigrams("hello world hahah hahah hahahhhahaha I am Christopher"))

#loading the reviews as input
pos_reviews = [line.rstrip('\n') for line in open('input/rt-polarity.pos', encoding='latin-1')]
neg_reviews = [line.rstrip('\n') for line in open('input/rt-polarity.neg', encoding='latin-1')]

pos_tokens = [generate_unigrams(line) for line in pos_reviews]
neg_tokens = [generate_unigrams(line) for line in neg_reviews]
dimension_y = len(pos_tokens) + len(neg_tokens)


# In[9]:


token_freq_dict={} # token frequency
for line in pos_tokens:
    for token in line:
        if token not in token_freq_dict:
            token_freq_dict[token] = 1
        else:
            token_freq_dict[token] += 1
            
for line in neg_tokens:
    for token in line:
        if token not in token_freq_dict:
            token_freq_dict[token] = 1
        else:
            token_freq_dict[token] += 1

####VERY IMPORTANT THRESHOLD FOR INFREQUENT WORDS###
infreq=[]
for key in token_freq_dict:
    if token_freq_dict[key] < 2:
        infreq.append(key)

for key in infreq:
    del token_freq_dict[key]
# #################################################
        
token_dict={}
counter = 0 # count to offset
pos_infreq=[]
neg_infreq=[]

for line in pos_tokens:
    for token in line:
        if token not in token_dict and token in token_freq_dict:
            token_dict[token] = counter
            counter = counter + 1
        if token not in token_freq_dict:
            pos_infreq.append(token)
            

for line in neg_tokens:
    for token in line:
        if token not in token_dict and token in token_freq_dict:
            token_dict[token] = counter
            counter = counter + 1
        if token not in token_freq_dict:
            neg_infreq.append(token)
            
##### Update tokens - remove infrequent tokens from the dict
for word in infreq:
    for line in pos_tokens:
        for token in line:
            if token == word:
                line.remove(token)
    for line in neg_tokens:
        for token in line:
            if token == word:
                line.remove(token)

for word in pos_infreq:
    for line in pos_tokens:
        for token in line:
            if token == word:
                line.remove(token)
    for line in neg_tokens:
        for token in line:
            if token == word:
                line.remove(token)
                
for word in neg_infreq:
    for line in pos_tokens:
        for token in line:
            if token == word:
                line.remove(token)
    for line in neg_tokens:
        for token in line:
            if token == word:
                line.remove(token)


def normalize(x):
    return x / x.sum()

def vectorize(tokens,groundtruth):
    # one vector may look like (x,x,x,x,......,1 or 0)
    vector = np.zeros(1 + len(token_dict))
    
    for token in tokens:
        vector[token_dict[token]] += 1
    
    vector = normalize(vector)
    vector[-1] = groundtruth
    
    return vector


# In[10]:


dimension_x = 1 + len(token_dict)
final_matrix = np.zeros((dimension_y,dimension_x))
# print(dimension_y)
# print(dimension_x)


# In[11]:


counter = 0 # again, count to offset
# now fill in the vectors into the matrix
# 1 for positive reviews
# 0 for negative reviews
for tokens in pos_tokens:
    final_matrix[counter,:] = vectorize(tokens,1)
    counter = counter + 1
for tokens in neg_tokens:
    final_matrix[counter,:] = vectorize(tokens,0)
    counter = counter + 1
#print(len(final_matrix))


# In[12]:


np.random.shuffle(final_matrix)
# preprocess to handle NaN -> Zero
where_are_NaNs = np.isnan(final_matrix)
final_matrix[where_are_NaNs] = 0


# In[13]:


# training / test set separation
np.random.shuffle(final_matrix)
label_matrix = final_matrix[:,-1]
feature_matrix = final_matrix[:,:-1]

# training_feature = feature_matrix[:int(len(final_matrix)*0.7),]
# training_groundtruth = label_matrix[:int(len(final_matrix)*0.7),]
# test_feature = feature_matrix[int(len(final_matrix)*0.7):,]
# test_groundtruth = label_matrix[int(len(final_matrix)*0.7):,]

training_feature = feature_matrix[:int(len(final_matrix)*0.8),]
training_groundtruth = label_matrix[:int(len(final_matrix)*0.8),]
test_feature = feature_matrix[int(len(final_matrix)*0.8):,]
test_groundtruth = label_matrix[int(len(final_matrix)*0.8):,]

# training_feature = feature_matrix[:int(len(final_matrix)*0.9),]
# training_groundtruth = label_matrix[:int(len(final_matrix)*0.9),]
# test_feature = feature_matrix[int(len(final_matrix)*0.9):,]
# test_groundtruth = label_matrix[int(len(final_matrix)*0.9):,]

#print(len(test_groundtruth))


# In[14]:


# prediction

def LogisticRegressionClf(x_train, y_train):
	clf = LogisticRegression(penalty='l2')
	clf.fit(x_train, y_train)
	return clf

def SVMClf(x_train, y_train):
	clf = SVC(kernel='linear')
	clf.fit(x_train, y_train)
	return clf 

def NaiveBayesClf(x_train, y_train):
	clf = MultinomialNB()
	clf.fit(x_train, y_train)
	return clf


#BASELINE: pure guessing
rand_state = 180
random.seed(rand_state)
success_count = 0
for result in test_groundtruth:
	random_predict = random.randint(0,1)
	if(random_predict == result):
		success_count += 1
        

print("Random Guessing: {}".format((success_count * 1.0) / len(test_groundtruth)))

model1 = LogisticRegressionClf(training_feature,training_groundtruth)
print("LR Classification Acc:", model1.score(test_feature, test_groundtruth))
label_pred = model1.predict(test_feature)
print(confusion_matrix(test_groundtruth, label_pred))

model2 = SVMClf(training_feature,training_groundtruth)
print("SVM Classification Acc:", model2.score(test_feature, test_groundtruth))
label_pred = model2.predict(test_feature)
print(confusion_matrix(test_groundtruth, label_pred))

model3 = NaiveBayesClf(training_feature,training_groundtruth)
print("NB Classification Acc:", model3.score(test_feature, test_groundtruth))
label_pred = model3.predict(test_feature)
print(confusion_matrix(test_groundtruth, label_pred))

### This part is for cross-validation: THIS PART IS COMMENTED OUT BECAUSE THIS TAKES REALLY LONG TO GENERATE A RESULT.
### I implemented this for practice (i know there's packages around)
# def evaluation(prediction: np.ndarray, groundtruth: np.ndarray):
#     # sanity check
#     if len(prediction) != len(groundtruth):
#         raise TypeError
    
#     tn,fp,fn,tp = 0,0,0,0 #true negative, false positive, false negative, true positive
    
#     for i in range(len(prediction)):
#         if prediction[i] == 0 and groundtruth[i] == 0:
#             tn += 1
#         if prediction[i] == 1 and groundtruth[i] == 0:
#             fp += 1
#         if prediction[i] == 0 and groundtruth[i] == 1:
#             fn += 1
#         if prediction[i] == 1 and groundtruth[i] == 1:
#             tp += 1
#     return tn,fp,fn,tp


# ################ This is the function to call for "Accuracy"############
# def accuracy(prediction: np.ndarray, groundtruth: np.ndarray):
#     tn,fp,fn,tp = evaluation(prediction,groundtruth)
#     return 1.0*(tp+tn)/(tp+tn+fp+fn)


# def merge_chunks(data_split,indices):
#     indices = list(indices).sort()
#     if len([indices]) < 2:
#         return data_split[0]
#     data_merged = data_split[indices[0]]
#     indices.remove(indices[0]) #remove the first element so that it does not get re-merged
#     for i in indices:
#         data_merged = np.concatenate(data_merged,data_split[i],axis=0)
        
#     return data_merged
        

# def cross_validation(model,x: np.ndarray,y: np.ndarray, k: int):
    
#     data = np.zeros((len(x),len(x[0])+1))
#     #combine and save to "data"
#     for i in range(len(x)):
#         data[i] = np.append(x[i],[y[i]])
#     # print(data)
#     np.random.shuffle(data)
#     data_split = np.array_split(data,k)
#     indices = set(range(k)) # a set containing 0 to k-1
#     acc_list = [] # the list containing all the output accuracies by k folds
#     for fold in range(k):
#         # merge the numpy arrays except for the validation set for training
#         other_indices = indices - set([fold])
#         training_set = merge_chunks(data_split,other_indices)
#         test_set = data_split[fold]
#         x_train = training_set[:,:-1]
#         y_train = training_set[:,-1]
#         x_test = test_set[:,:-1]
#         y_test = test_set[:,-1]
        
#         model.fit(x_train,y_train)
#         y_prediction = model.predict(x_test)
        
#         acc_list.append(accuracy(y_prediction,y_test))
#     return sum(acc_list) / len(acc_list)


# The best result
# Random Guessing: 0.49132676980778245
# D:\PROGRAMS\Anaconda\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
#   FutureWarning)
# LR Classification Acc: 0.7191748710736052
# [[770 271]
#  [328 764]]
# SVM Classification Acc: 0.726207219878106
# [[775 266]
#  [318 774]]
# NB Classification Acc: 0.7744960150023441
# [[842 199]
#  [282 810]]


# In[ ]:




