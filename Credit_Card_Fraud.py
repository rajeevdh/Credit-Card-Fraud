import sys
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns


def versions():
    print('Python : {}'.format(sys.version))
    print('Numpy : {}'.format(np.__version__))
    print('Pandas : {}'.format(pd.__version__))
    print('scipy : {}'.format(scipy.__version__))
    print('sklearn : {}'.format(sklearn.__version__))
    print('Seaborn : {}'.format(sns.__version__))

def Explore_Data(data_set):
    print('--Columns--')
    print(data_set.columns)       # Display all the features in the data
    print('--Shape--')
    print(data_set.shape)         # Dimensions of the data
    print('--Describe--')
    print(data_set.describe)      # Mean, Median, Mode etc.

def Histogram_plt(data_set):
    data_set.hist(figsize=(20, 20))
    plt.savefig('C:\\Users\\Soujanya\\Desktop\\Rajeev\\DataSets_ML\\creditcardfraud\\hist.png')
    print('Image Saved')

versions()

data = pd.read_csv('C:\\Users\\Soujanya\\Desktop\\Rajeev\\DataSets_ML\\creditcardfraud\\creditcard.csv')
#Explore_Data(data)

#Down sample data due to computational limitations
data_fraud = data[data['Class'] == 1].sample(frac=1, random_state=1)  # Returns all fraud data values
data_Valid = data[data['Class'] == 0].sample(frac=0.2, random_state=1)  # returns 10% of valid-transaction data values

data_all = data_Valid
data_all = data_all.append(data_fraud, ignore_index=True)

print('---------------Explore Data_Fraud-------------------')
#Explore_Data(data_fraud)
print('Shape : {}'.format(data_fraud.shape))

print('---------------Explore Data_Valid-------------------')
#Explore_Data(data_Valid)
print('Shape : {}'.format(data_Valid.shape))

print('---------------Explore Data_All-------------------')
#Explore_Data(data_all)
print('Shape : {}'.format(data_all.shape))

data_all.sample(frac=1)

#Plot Histogram of each parameter
Histogram_plt(data_all)

Outlier_fraction = len(data_fraud) / float(len(data_Valid))

print('Outlier_fraction : {}'.format(Outlier_fraction))
print('Fraud Transactions : {}'.format(len(data_fraud)))
print('Valid Transactions : {}'.format(len(data_Valid)))


#Get all the columns from the dataframes
columns = data_all.columns.tolist()

#Filter data to seperate features and class.
columns = [c for c in columns if c not in ['Class']]

print(columns)

#fetch data values and Class
X = data_all[columns]
Y = data_all['Class']


print('---X ---')
print(X.shape)
print('--Y--')
print(Y.shape)

#define a random state
state = 1

#Define outlier detection methods
classifiers = {
            'Isolation Forest': IsolationForest( max_samples = len(X), contamination= Outlier_fraction, random_state= state),
             'Local Outlier Factor':LocalOutlierFactor(n_neighbors=20, contamination=Outlier_fraction)
}

#Fit Model
n_outliers = len(data_fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name =='Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred= clf.predict(X)

    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1

    n_errors = (y_pred !=Y).sum()

    print('----Printing Stats')
    print('{} : {}'.format(clf_name,n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y,y_pred))