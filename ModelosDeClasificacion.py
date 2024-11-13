
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib
import pyodbc
import pandas as pd
import requests
import pandas as pd
import subprocess
import json
import sqlite3
import psycopg2

#Establecer la conexión
conn = sqlite3.connect("")
db =pd.read_sql("SELECT * FROM ENTRENAMIENTO_VITALIA", conn)
"""#DEFINICIÓN DE VARIABLES


"""


# Definición de variables
X = df.drop(['TRATAMIENTO', 'DESCRIPCION'], axis=1)
y = df['TRATAMIENTO']
X

y

"""#ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN"""

# División de datos de entrenamiento y testeo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Balanceo de variable respuesta
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

"""#RANDOMFOREST"""

# RandomForest
rf_model = RandomForestClassifier(max_depth=5, max_features='sqrt', min_samples_leaf=3, n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)

"""#LGB"""

# LGBM
lgb_model = LGBMClassifier(random_state=42, verbose=-1)
lgb_model.fit(X_resampled, y_resampled)

"""#XGB"""

# XGB
xgb_model = XGBClassifier(random_state=42, verbosity=0)
xgb_model.fit(X_resampled, y_resampled)

"""#CONSOLIDACIÓN DE MODELOS"""

# Guardar los modelos entrenados
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(lgb_model, 'lgbm_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')

# Cargar los modelos guardados
rf_model = joblib.load('random_forest_model.pkl')
lgb_model = joblib.load('lgbm_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

# Predicciones de los modelos
y_pred_rf = rf_model.predict(X_test)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Predicción promedio
y_pred_avg = np.round((y_pred_rf + y_pred_lgb + y_pred_xgb) / 3).astype(int)

"""#MÉTRICAS DE CADA MODELO"""

print("\nResumen del modelo:")
print("Accuracy con Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix con Random Forest:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report con Random Forest:\n", classification_report(y_test, y_pred_rf))

print("Accuracy con LightGBM:", accuracy_score(y_test, y_pred_lgb))
print("Confusion Matrix con LightGBM:\n", confusion_matrix(y_test, y_pred_lgb))
print("\nClassification Report con LightGBM:\n", classification_report(y_test, y_pred_lgb))

print("Accuracy con XGBoost:", accuracy_score(y_test, y_pred_xgb))
print("Confusion Matrix con XGBoost:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nClassification Report con XGBoost:\n", classification_report(y_test, y_pred_xgb))

print("Accuracy con predicción promedio:\n", accuracy_score(y_test, y_pred_avg))
print("\nConfusion Matrix con predicción promedio:\n", confusion_matrix(y_test, y_pred_avg))
print("\nClassification Report con predicción promedio:\n", classification_report(y_test, y_pred_avg))

"""#EJEMPLO DE CLASIFICACIÓN DE 3 PACIENTES

"""

new_data = pd.DataFrame({
    "INSUFICIENCIA_CARDIACA": [1, 1, 0],
    "HIPERTENSION": [1, 0, 1],
    "MAS75": [0, 1, 0],
    "DIABETES_MELLITUS": [0, 0, 1],
    "DERRAME": [1, 1, 0],
    "TIA_TROMBOEMBOLIA": [0, 0, 0],
    "ENFERMEDAD_VASCULAR": [0, 0, 1],
    "EDAD65A74": [0, 0, 1],
    "GENERO_FEMENINO": [1, 0, 0],
    "FUNCION_RENAL_ANORMAL": [0, 1, 1],
    "FUNCION_HEPATICA_ANORMAL": [0, 1, 0],
    "HISTORIA_DISP_SANGRADO": [1, 0, 0],
    "INR": [0, 2, 0],
    "MEDICAMENTOS_ACTUAL": [0, 0, 1],
    "CONSUMO_ALCOHOL": [0, 1, 1],
    "WARFARIN": [1, 0, 0],
    "SANGRADO_ABUNDANTE": [1, 0, 1],
    "CREATININA": [0, 1, 0],
    "ANEMIA": [0, 1, 1],
    "CANCER": [0, 0, 1],
    "EMBOLIA_PULMONAR": [0, 1, 0]
})

newdata = calcular_puntajes(new_data)

# Preprocesar los nuevos datos
new_data_scaled = scaler.transform(new_data)

# Hacer la predicción con cada modelo
pred_rf = rf_model.predict(new_data_scaled)
pred_lgb = lgb_model.predict(new_data_scaled)
pred_xgb = xgb_model.predict(new_data_scaled)

# Promediar las predicciones
pred_avg = np.round((pred_rf + pred_lgb + pred_xgb) / 3).astype(int)

# Mostrar las predicciones y descripciones para nuevos datos
for i, pred in enumerate(pred_avg):
    treatment = tratamientos[pred]
    description = descripciones[treatment]
    print(f"\nPaciente {i + 1}:")
    print(f"Tratamiento recomendado: {treatment}")
    print(f"Descripción: {description}")

# Adicionalmente, imprimir las predicciones individuales para cada modelo
print("\nPredicciones individuales para cada modelo:")
for i in range(len(new_data)):
    print(f"\nPaciente {i + 1}:")
    print(f"Predicción Random Forest: {tratamientos[pred_rf[i]]}")
    print(f"Predicción LightGBM: {tratamientos[pred_lgb[i]]}")
    print(f"Predicción XGBoost: {tratamientos[pred_xgb[i]]}")

