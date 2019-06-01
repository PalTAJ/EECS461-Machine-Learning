import cPickle as pickle
import  numpy as np
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn import svm

# -*- coding: utf-8 -*-


#########
# PART a)Feature Extraction
def feature1(x):
    """This feature computes the proportion of black squares to the
       total number of squares in the grid.
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature1_value: type-float
       """
    feature1_value=0.0
    black =0;allcolors =0
    for i in range(0,len(x),1):
        for j in range(0,len(x[i]),1):
            if x[i][j] ==1:

                black +=1.0 ; allcolors+=1.0
            else:
                allcolors+=1.0
        feature1_value = (black/allcolors)
    return feature1_value
# print feature1(train_positives[0])
# fn1 = feature1(train_negatives[0])
###########################



def feature2(x):
    """This feature computes the sum of the max of continuous black squares
       in each row
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature2_value: type-float
       """
    feature2_value=0.0
    count = 0
    for i in range(0,len(x),1):
        feature2_value += checker(x[i])
    return feature2_value

# train_positives = pickle.load(open('training_set_positives.p', 'rb'))
# fn2 = feature2(train_positives[0])
# print fn2

#########################################
###### part b
#########################################
# PART b) Preparing Data
def part_b():
    X1 = [];y1 =[]
    train_positives = pickle.load(open('training_set_positives.p', 'rb'))
    train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))
    for p in range(0,len(train_positives),1):
        f1p = feature1(train_positives[p])
        f2p = feature2(train_positives[p])
        X1.append([f1p,f2p])
        y1.append(1)
    for n in range(0,len(train_negatives),1):
        f1n = feature1(train_negatives[n])
        f2n = feature2(train_negatives[n])
        X1.append([f1n,f2n])
        y1.append(0)
    X = np.array(X1)
    Y = np.array(y1)


    return [X,Y]
# print part_b()
#
# PART c) Classification with SGDClassifier
def part_c(x, alpha=0.001, max_iter=20, random_state=0):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    X = part_b()[0]
    Y = part_b()[1]
    clf = linear_model.SGDClassifier(alpha=0.001,max_iter=20,random_state=0)
    clf.fit(X, Y)
    # print x
    predicted_class = clf.predict([x])
    return predicted_class

# x2 = part_b()[0]
# for val in x2:
#     print part_c(val)



# # PART d) Assess the performance of the classifier in part c
def part_d():  #     # https://www.kdnuggets.com/faq/precision-recall.html
    pred = [];ylabel =[];c=0;cmatch=0
    x2 = part_b()[0] ; y2 = part_b()[1]
    for val in x2:
        pred.append(part_c(val)[0])
    for val2 in y2:
        ylabel.append(val2)
    cm = confusion_matrix(ylabel, pred)
    precision = precision_score(ylabel, pred, average='weighted')
    recall = recall_score(ylabel, pred, average='weighted')
    # for i in range(0,len(ylabel)-1,1):
    #     if ylabel[i] ==1 and ylabel[i]==pred[i]:
    #         cmatch+=1
    #     if ylabel[i]==1 and ylabel[i]!= pred[i]: #######################################################
    #         c+=1
    # recall = cmatch
    # t  = c/100.0; t2 = cmatch/t
    # precision = t2
    # # print recall
    # # print precision
    confusion_matrixx = cm
    return [precision, recall, confusion_matrixx]
# print part_d()


# PART e) Classification with RandomForestClassifier
def part_e(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    X = part_b()[0];Y = part_b()[1]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, Y)
    predicted_class = clf.predict([x])
    return predicted_class

# x2 = part_b()[0]
# for val in x2:
#     print part_e(val)


# PART f) Assess the performance of the classifier in part e
def part_f():
    pred = [];ylabel =[];c=0;cmatch=0
    x2 = part_b()[0] ; y2 = part_b()[1]
    for val in x2:
        pred.append(part_e(val)[0])
    for val2 in y2:
        ylabel.append(val2)
    cm = confusion_matrix(ylabel, pred)
    precision = precision_score(ylabel, pred, average='weighted')
    recall = recall_score(ylabel, pred, average='weighted')
    # for i in range(0,len(ylabel)-1,1):
    #     if ylabel[i] ==1 and ylabel[i]==pred[i]:
    #         cmatch+=1
    #     if ylabel[i]==1:
    #         c+=1
    # recall = cmatch
    # t  = c/100.0; t2 = cmatch/t
    # precision = t2
    # # print recall
    # # print precision
    confusion_matrixx = cm
##################
    return [precision, recall, confusion_matrixx]
# print part_f()



# PART g) Your Own Classification Model
def custom_model(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    X = part_b()[0];Y = part_b()[1]
    clf = svm.SVC()
    clf.fit(X, Y)
    predicted_class = clf.predict([x])
    return predicted_class
# x2 = part_b()[0]
# for val in x2:
#     part_e(val)


#######################

# def part_final():
#     pred = [];ylabel =[];c=0;cmatch=0
#     x2 = part_b()[0] ; y2 = part_b()[1]
#     for val in x2:
#         pred.append(custom_model(val)[0])
#     for val2 in y2:
#         ylabel.append(val2)
#     cm = confusion_matrix(ylabel, pred)
#     precision = precision_score(ylabel, pred, average='weighted')
#     recall = recall_score(ylabel, pred, average='weighted')
#
#     # for i in range(0,len(ylabel)-1,1):
#     #     if ylabel[i] ==1 and ylabel[i]==pred[i]:
#     #         cmatch+=1
#     #     if ylabel[i]==1:
#     #         c+=1
#     # recall = cmatch
#     # t  = c/100.0; t2 = cmatch/t
#     # precision = t2
#     # # print recall
#     # # print precision
#
#     confusion_matrixx = cm
# ##################
#     return [precision, recall, confusion_matrixx]
# print part_final()