# Author: Preet Shah
# Date: March 1, 2023
# Description: ML Classifiers (Bayes vs SVM)
# Revision History
# Name      Date        Description
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
# scl       3/01/2023   Initial Code Base for Students in class
# scl       3/03/2023   Updated with metrics for accuracy score
# scl       3/04/2023   Add code for label encoder
# scl       3/05/2023   Add SVM and Decision Trees Classifier
# scl       3/05/2023   Refactor the to be more procedural
# scl       3/05/2023   Mount HU Google Drive with the data file(s)

'''import os
from google.colab import drive
drive.mount('/content/drive')

location ='drive/MyDrive/hofstra/hit_data/hit215/'
list_of_files = os.listdir(location)
list_of_files'''

import numpy as np
import pandas as pd
import seaborn as sns

# Let us start using Sci-Kit Learn Modules for Machine Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def loadData():
  # load the raw data into a pandas dataframe
  df = pd.read_csv(r'C:/Users/19086/Desktop/HIT 215/New folder/IMR.csv')

  # label encode the object to numeric
  le = LabelEncoder()
  df.MaternalEd                   = le.fit_transform(df.MaternalEd)
  df.MHC                         = le.fit_transform(df.MHC)
  df.MN                           = le.fit_transform(df.MN)
  df.SU                           = le.fit_transform(df.SU)
  df.MG                           = le.fit_transform(df.MG)
  df.SS                          = le.fit_transform(df.SS)
 
 
  return df

def createTrainTestSplit(df_IMR):
  # change the random_state each time we sample
  # the dataframe, just another way to randomize
  from datetime import datetime
  random_state = datetime.now()
  random_state = random_state.second

  # values for test and training sets
  pct_train = 0.70
  pct_test  = 1.00 - pct_train

  # get a training sample
  df_IMR_train = df_IMR.sample(frac=pct_train, random_state=random_state)

  # use python slicing to do the trick 
  df_IMR_train_X = df_IMR_train.iloc[: , : -1]
  df_IMR_train_y = df_IMR_train.iloc[: ,-1 :]

  # get a testing sample
  df_IMR_test = df_IMR.sample(frac=pct_test, random_state=random_state)

  # use python slicing to do the trick 
  df_IMR_test_X = df_IMR_test.iloc[: , : -1]
  df_IMR_test_y = df_IMR_test.iloc[: ,-1 :]

  # not a necessary step but just for teaching purposes
  X_train = np.array(df_IMR_train_X.values)
  y_train = np.array(df_IMR_train_y.values)

  X_test = np.array(df_IMR_test_X.values)
  y_test = df_IMR_test_y.values

  return X_train, y_train, X_test, y_test

# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
#               Begin - Section for Model Functions 
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
def createModelNB(X_train, y_train):
  model = GaussianNB()
  y_train.reshape(len(y_train))
  model.fit(X_train, y_train.ravel())   # use ravel() to get ride of 1D warning. 
  return model

def createModelSVM(X_train, y_train):
  model = SVC()
  y_train.reshape(len(y_train))
  model.fit(X_train, y_train.ravel())   # use ravel() to get ride of 1D warning. 
  return model

def createModelDT(X_train, y_train):
  model = DecisionTreeClassifier(random_state=42)
  y_train.reshape(len(y_train))
  model.fit(X_train, y_train.ravel())   # use ravel() to get ride of 1D warning. 
  return model
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -
#               End - Section for Model Functions 
# -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

def predictModel(model_IMR, X_test, y_test):
  y_pred = model_IMR.predict(X_test)
  y_pred.reshape(len(y_test),1)
  return y_pred

def modelPerformance(y_pred, y_true):
  performance = accuracy_score(y_pred, y_true)
  return performance

def main():
  # step 1. Load the data file into a data frame
  data = loadData()

  # step 2. Split the set into test & training sets
  X_train, y_train, X_test, y_test = createTrainTestSplit(data)

  # step 3. Model, train, test --> performance
  # -   -   -   Bayes (NB)   -   -   -   -   -
  model_bayes   = createModelNB(X_train, y_train)
  predict_bayes = predictModel(model_bayes, X_test, y_test)
  perf_bayes    = modelPerformance(predict_bayes, y_test)

  # -   -   -   SVM (Classifier)   -   -   -   -   -
  model_svm   = createModelSVM(X_train, y_train)
  predict_svm = predictModel(model_svm, X_test, y_test)
  perf_svm    = modelPerformance(predict_svm, y_test)

  # -   -   -   -   -   -   -   -   -   -   -   -   -   -   -   -

 
  res = np.array([perf_bayes, perf_svm])

  return res

from time import sleep
for i in range(5):
  results = main()
  print(results)
  sleep(3)
