import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

# 1. Identify and characterize a dataset

Data = pd.read_csv(r'D:\CNG514_Spring_2021\CNG514-Assignment-2_spring22\CNG514-Assignment-2-data.csv')
size = Data.size
shape = Data.shape
print('Size={}\n shape={}'.format(size,shape))
print(Data)
Data.info()

# -----------------------------------------------------------------------------------------------
# Identify and characterize attributes

# Central Tendency Measures
print("Mean of the 'AGE', Which is a Continuous Attribute")
print(Data['AGE'].mean())

print('Mode of the Data')
cols=[]
data_mode = Data.iloc[:, 2:14]
print(data_mode.mode())


# Data preprocessing

print(Data.isnull())
print(Data.notnull())
print("The missing values in each attributes:")

print(Data.isna().sum())


print('Identification of Noisy Data')

print(Data.max())
print(Data.min())


Data['GENDER'] = Data['GENDER'].map({'M': 2, 'F': 1})

label_encoder = preprocessing.LabelEncoder()
Data['LUNG_CANCER'] = label_encoder.fit_transform(Data['LUNG_CANCER'])
# print(Data)

# Let's create separate dataframe for features and target
features = Data.drop(['LUNG_CANCER'], axis = 1)
target = Data['LUNG_CANCER']
print(features)
print(target)

# Normalization

scaler = MinMaxScaler()
scaler.fit(features)
scaled = scaler.fit_transform(features)
features = pd.DataFrame(scaled, columns=features.columns)
print(features)



# KNN Classifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25)

classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Confusion Matrix and Classification Report

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Calculating error for K values between 1 and 15
error = []

for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 15), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()


# Cross Validation Score after 10 folds

knn_cv = KNeighborsClassifier(n_neighbors=11)
cv_scores = cross_val_score(knn_cv, features, target, cv=10)
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))

# cross Validation mean Score of "f1", "precision", "recall", "accuracy" after 10 folds.
for score in ["f1", "precision", "recall", "accuracy"]:
        cvs = cross_val_score(knn_cv, features, target, scoring=score, cv=10).mean()
        print(score + " : "+ str(cvs))



# Confusion Matrix for Each fold and Their Average Confusion Matrix

confusion_matrix_list = []
kf = KFold(n_splits=10)
w = kf.split(features)
for train_ind, test_ind in w:

    feat_train, feat_test = features.iloc[train_ind], features.iloc[test_ind]
    targ_train, targ_test = target.iloc[train_ind], target.iloc[test_ind]

    knn_cv.fit(feat_train, targ_train)
    conf_mat = confusion_matrix(targ_test, knn_cv.predict(feat_test))
    print(conf_mat)
    confusion_matrix_list.append(conf_mat)


mean_confusion_matrix_list = np.mean(confusion_matrix_list, axis=0)
print(mean_confusion_matrix_list)


