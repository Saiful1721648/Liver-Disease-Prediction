from django.shortcuts import render
import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    patients = pd.read_csv('indian_liver_patient.csv')
    nan_raws = patients[patients['Albumin_and_Globulin_Ratio'].isnull()]
    patients['Dataset'].value_counts()
    patients = patients.dropna()
    X = patients.iloc[:, 0:10]
    y = patients.iloc[:, -1]
    X.Gender = X.Gender.map({'Male': 0, 'Female': 1})
    patients.Gender = patients.Gender.map({'Male': 0, 'Female': 1})
    df_majority = patients[patients.Dataset == 1]
    df_minority = patients[patients.Dataset == 2]
    df_minority_upsamples = resample(df_minority, replace=True, n_samples=414, random_state=123)
    df_upsampled = pd.concat([df_majority, df_minority_upsamples])
    X = df_upsampled.iloc[:, 0:10]
    y = df_upsampled.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    pr = SVC(gamma='auto')
    pr.fit(X_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    val9 = float(request.GET['n9'])
    val10 =float(request.GET['n10'])

    pred = pr.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9,val10]])
    result1 = ""

    if pred == [1]:
        result1 = "postive"
    else:
        result1 = "negative"

    return render(request, 'predict.html', {"result2": result1})
