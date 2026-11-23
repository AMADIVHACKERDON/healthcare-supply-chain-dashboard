import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn import svm
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model,preprocessing
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans

digits = load_digits()
data = scale(digits.data)
y = digits.target

k = len(np.unique(y))
samples , features = data.shape

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name,estimator.inertia_,
             metrics.homogeneity_score(y,estimator.labels_),
             metrics.completeness_score(y,estimator.labels_),
             metrics.v_measure_score(y,estimator.labels_),
             metrics.adjusted_rand_score(y,estimator.labels_),
             metrics.adjusted_mutual_info_score(y,estimator.labels_),
             metrics.silhouette_score(data,estimator.labels_,metric='euclidean')))

clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)


learn = datasets.load_breast_cancer()
df_breast_cacer = pd.DataFrame(learn.data, columns=learn.feature_names)

print(df_breast_cacer)
print(learn.feature_names)
print(learn.target_names)

a = learn.data
b = learn.target

x_train,  x_test,y_train, y_test = sklearn.model_selection.train_test_split(a,b, test_size = 0.1)
print(x_train,y_train)
classes = ['malignant' 'benign']
clf = svm.SVC(kernel="linear", C=1)
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc)

data = pd.read_csv("dataset for AI.csv.csv", sep=",")
print(data.head())


le = preprocessing.LabelEncoder()


weight_kg = le.fit_transform(list(data["weight in kg"]))
cm = le.fit_transform(list(data["height in cm"]))
gender = le.fit_transform(list(data["gender"]))

X = list(zip(weight_kg,cm))
Y = list(gender)

x_train, x_test,y_train, y_test = train_test_split(X,Y,test_size= 0.1)

print(x_test,y_test)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
bcc = model.score(x_test,y_test)
print(bcc)

predicted = model.predict(x_test)
names = ["unacc","acc","good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ",names[predicted[x]],"Data: ", x_test[x], "Actual:", names[y_test[x]])
    n = model.kneighbors([x_test[x]],5,True)
    print(n)


