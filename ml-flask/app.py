from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Crear la aplicación Flask
app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('/workspaces/ml-flask/ml-flask/models/model.pkl')

# Ruta principal
@app.route('/')
def home():
    return render_template('/workspaces/ml-flask/ml-flask/templates/index.html')

# Ruta para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        features = request.form['features']
        # Convertir los datos en un array de numpy
        features = np.array([float(x) for x in features.split(',')]).reshape(1, -1)
        # Realizar la predicción
        prediction = model.predict(features)
        species = ["Setosa", "Versicolor", "Virginica"][prediction[0]]
        # Retornar el resultado
        return render_template('index.html', prediction=species)
    except Exception as e:
        return render_template('index.html', prediction="Error: Verifique los datos ingresados.")

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

