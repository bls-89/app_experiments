from flask import Flask, render_template, request
import keras
import numpy as np

app = Flask(__name__)
model_path = '777'
model = keras.models.load_model(model_path)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0][0], 2)


    return render_template('index.html', prediction_text='Predicted Variable Value: {}'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0')
