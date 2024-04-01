#Время 26:21
import click
from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
app = Flask(__name__)

@app.route('/')
@app.route('/home')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    with open("../model/lr_models.pkl", "rb") as f:
        lr_models = pickle.load(f)
    with open("../model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    values = list()
    keys = list()
    df_cat = pd.DataFrame()
    for key, val in request.form.items():
        values.append(val)
        keys.append(key)
    df_cat = pd.DataFrame([values], columns=keys)

    # Преобразование категориальных значений в колонках
    df_cat["gender"] = df_cat["gender"].map({"male": 0, "female": 1})
    df_cat["race/ethnicity"] = df_cat["race/ethnicity"].map({"group A": 0, "group B": 1,
                                                             "group C": 2, "group D": 3,
                                                             "group E": 4, })

    df_cat["lunch"] = df_cat["lunch"].map({"standard": 0, "free/reduced": 1})
    df_cat["test preparation course"] = df_cat["test preparation course"].map({"none": 0, "completed": 1})
    categorical_cols = df_cat.select_dtypes('object').columns.tolist()
    with open("../model/encoded_cols.pkl", "rb") as f:
        encoded_cols = pickle.load(f)
    df_cat[encoded_cols] = encoder.transform(df_cat[categorical_cols])
    df_cat.drop(labels = ["parental level of education"], axis = 1, inplace = True)

    #Вычисления модели по полученным данным
    scores = list()
    for i in lr_models:
        scores.append(i.predict(df_cat))

    return render_template('form.html', final_grade=scores)


