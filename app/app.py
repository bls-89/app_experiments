# -*- coding: utf-8 -*-
"""app_vkr_last.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PmKZNUmzewNrGO5q3Qhl1frOWbQdVKrP
"""

# Импорт библиотек
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template

app = Flask(__name__)

# Загрузка модели и передача в функцию параметров  -  свойств 

def get_prediction(property1, property2, property3, property4, property5, property6, property7, property8, property9, property10, property11, property12):

    model = keras.models.load_model("/content/models/matrica_napolnitel_best_params_mlp")
    pred = model.predict([property1, property2, property3, property4, property5, property6, property7, property8, property9, property10, property11, property12])

    return pred[0][0]

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/', methods=['post', 'get'])

def calc_prediction():
    properties_list = []
    message = ''
    if request.method == 'POST':
        
       #  данные из заполненной формы поступают в список, который затем передадается функции get_prediction
        for i in range(1,13,1):
            property = request.form.get(f'property{i}')
            properties_list.append(float(property))
            
        message = get_prediction(*properties_list)

    # рендеринг шаблона и вывод сообщения с предсказанием    
    return render_template("index.html", message=message)

# Запуск приложения  
if __name__ == '__main__':
  app.run()
