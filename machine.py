import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Leer el conjunto de datos y cargarlo en un DataFrame de pandas
diabetes_df = pd.read_csv('diabetes.csv')

# Seleccionar las columnas a utilizar
lista_caract = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_df[lista_caract]

# Seleccionar la columna de etiquetas ('Outcome')
lista_etiq = ['Outcome']
y = diabetes_df[lista_etiq]

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Definir el modelo SVM con kernel lineal
clf = SVC(kernel='linear')

# Entrenar el modelo y medir el tiempo de entrenamiento
hora_inicio_entrenamiento = time()
clf.fit(X_train, y_train.values.ravel())
tiempo_entrenamiento = time() - hora_inicio_entrenamiento

# Imprimir el tiempo tomado para el entrenamiento
print("Entrenamiento terminado en {} segundos".format(tiempo_entrenamiento))

# Realizar la predicción y medir el tiempo de predicción
hora_inicio_prediccion = time()
y_pred = clf.predict(X_test)
tiempo_prediccion = time() - hora_inicio_prediccion
# Imprimir el tiempo tomado para la predicción
print("Predicción terminada en {} segundos".format(tiempo_prediccion))
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo: {:.2f}%".format(precision * 100))

