#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import json
import gzip
import pickle
import os

# Cargar los archivos CSV comprimidos
train_path = "./files/input/train_data.csv.zip"
test_path = "./files/input/test_data.csv.zip"

df_train = pd.read_csv(train_path, compression="zip")
df_test = pd.read_csv(test_path, compression="zip")

# Agregar la columna 'Age' calculada a partir del año actual
current_year = 2021
df_train['Age'] = current_year - df_train['Year']
df_test['Age'] = current_year - df_test['Year']

# Eliminar columnas innecesarias
columns_to_remove = ['Year', 'Car_Name']
df_train.drop(columns=columns_to_remove, inplace=True)
df_test.drop(columns=columns_to_remove, inplace=True)

# Separar las características (X) y la variable objetivo (y)
X_train = df_train.drop('Present_Price', axis=1)
y_train = df_train['Present_Price']
X_test = df_test.drop('Present_Price', axis=1)
y_test = df_test['Present_Price']

# Identificar las variables categóricas y numéricas
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
numeric_features = [col for col in X_train.columns if col not in categorical_features]

# Crear un preprocesador para las características
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(), categorical_features),
        ('numeric', MinMaxScaler(), numeric_features)
    ]
)

# Crear el pipeline de modelo
model_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('feature_selection', SelectKBest(f_regression)),
    ('regression', LinearRegression())
])

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'feature_selection__k': np.arange(1, 12),
    'regression__fit_intercept': [True, False],
    'regression__positive': [True, False]
}

# Ejecutar búsqueda de hiperparámetros con validación cruzada
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

# Ajustar el modelo con los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Asegurarse de que el directorio para guardar el modelo exista
model_dir = 'files/models'
os.makedirs(model_dir, exist_ok=True)

# Guardar el modelo entrenado
model_file = os.path.join(model_dir, 'model.pkl.gz')
with gzip.open(model_file, 'wb') as f:
    pickle.dump(grid_search, f)

# Realizar predicciones con el modelo ajustado
train_predictions = grid_search.predict(X_train)
test_predictions = grid_search.predict(X_test)

# Calcular métricas de rendimiento
train_metrics = {
    'type': 'metrics',
    'dataset': 'train',
    'r2': float(r2_score(y_train, train_predictions)),
    'mse': float(mean_squared_error(y_train, train_predictions)),
    'mad': float(median_absolute_error(y_train, train_predictions))
}

test_metrics = {
    'type': 'metrics',
    'dataset': 'test',
    'r2': float(r2_score(y_test, test_predictions)),
    'mse': float(mean_squared_error(y_test, test_predictions)),
    'mad': float(median_absolute_error(y_test, test_predictions))
}

# Asegurarse de que el directorio para guardar las métricas exista
output_dir = 'files/output'
os.makedirs(output_dir, exist_ok=True)

# Guardar las métricas en formato JSON
metrics_file = os.path.join(output_dir, 'metrics.json')
with open(metrics_file, 'w', encoding='utf-8') as f:
    f.write(json.dumps(train_metrics) + '\n')
    f.write(json.dumps(test_metrics) + '\n')