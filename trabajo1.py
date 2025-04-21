# @title

# Se debe tener instalado las siguientes librerias 
# pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
import random
from funciones import find_s, aplicarHipotesis, evaluar_hipotesis, metricas, separarDatosEnConjuntos, entrenar_naive_bayes, predecir_naive_bayes

print("PUNTO 1")
# leemos los datos del csv y le indicamos que columnas vamos a utilizar
url = 'https://drive.google.com/uc?export=download&id=1BQEFonHa5aYO4MTg1EWxGIuRSgCw6ZXb'
# leemos los datos del csv y le indicamos que columnas vamos a utilizar
datosPrestamos = pd.read_csv(url, usecols=[0, 1, 2, 5,12,13], encoding='latin1')

# Filtrar solo personas de 50 años. 
# Estamos haciendo una copia exacta de la lista de personas. Esta generando una nueva referencia.
personasDe50Anios = datosPrestamos[datosPrestamos['Edad'] == 50].values.tolist().copy() 

entrenamiento, prueba = separarDatosEnConjuntos(personasDe50Anios,0.75)
########################

# Llamar a la funcion Find-S
hypothesis = find_s(entrenamiento)

# Imprimimos los índicides del csv (Edad, sexo....)
print("La hipótesis final es:")
#print(datosPrestamos.columns.to_list())
print(datosPrestamos.columns.to_list()[:len(datosPrestamos.columns) -1])
# Mostrar la hipótesis final
print( "  ",hypothesis[0],"    ",hypothesis[1],"               ",hypothesis[2],"                ",hypothesis[3],"                     ",hypothesis[4])
#print("La hipótesis final es:", hypothesis)

# Estamos controlando la cantidad de personas total, 
print("\nCantidades obtenidas luego de filtrar a las personas de 50")
print( {"Total de personas con 50 años:": len(personasDe50Anios), "Entrenamiento": len(entrenamiento),"Prueba":len(prueba)})
# Llamar a la funcion aplicarHipotesis
contador= aplicarHipotesis(prueba,hypothesis)
print("\nAplicamos la Hipótesis al grupo de prueba y obtuvimos estos resultados")
#print("PREDICCIÓN    ", contador,"  APROBADOS DE UN TOTAL DE ",len(prueba)  )
#print("EN PORCENTAJE    ", int(contador/len(prueba)*100) ," % "  )

print({
    "Prediccion": contador,
    "Aprobados de un total de": len(prueba),
    "Porcentaje": str(int(contador / len(prueba) * 100)) + " %"
})
############################################################
# PUNTO 2
print("\nPUNTO 2")
# Obtener los valores
TP, FP, TN, FN = evaluar_hipotesis(hypothesis, prueba)

metricas(TP,FP,TN,FN)


###################################################
# PUNTO 3
print("\nPUNTO 3")
# Paso 1: Filtrar personas entre 40 y 45 años
personas_40_45 = datosPrestamos[(datosPrestamos['Edad'] >= 40) & (datosPrestamos['Edad'] <= 45)].copy()

# Convertir todas las columnas a tipo string para convertirlo en categórico
for col in personas_40_45.columns:
    personas_40_45[col] = personas_40_45[col].astype(str)

# Dividir aleatoriamente los datos en entrenamiento (80%) y prueba (20%)
entrenamiento, prueba = separarDatosEnConjuntos(personas_40_45.values.tolist(),0.8)

# Paso 2: Implementar Naive Bayes manualmente

# Entrenar el modelo
conteo_clases, conteo_atributos = entrenar_naive_bayes(entrenamiento)

# Paso 3: Evaluar el modelo en el conjunto de prueba
TP = FP = TN = FN = 0
y_true = []
y_scores = []

def construirMatrizDeConfusion(prediccion):
    global TP, FP, TN, FN
    if prediccion == "OTORGADO":
        if verdadero_estado == "OTORGADO":
            TP += 1      #se etiquetó bien, la predicción fue correcta
        else:
            FP += 1      #está etiquetado como "OTORGADO" pero en realidad es "RECHAZADO"
    else:
        if verdadero_estado == "RECHAZADO":
            TN += 1      #se etiquetó bien, la predicción fue correcta ,está "RECHAZADO"
        else:
            FN += 1

for fila in prueba:
    prediccion, probabilidades = predecir_naive_bayes(fila, conteo_clases, conteo_atributos, len(entrenamiento))
    verdadero_estado = fila[-1]
    y_true.append(1 if verdadero_estado == "OTORGADO" else 0)
    y_scores.append(probabilidades["OTORGADO"] if "OTORGADO" in probabilidades else 0)

    construirMatrizDeConfusion(prediccion)

# USAMOS UNA FUNCION PARA EL PUNTO 2 y 3. Para la parte de metricas.

metricas(TP,FP,TN,FN)

# Paso 5: Graficar la curva ROC
def calcular_curva_roc(y_true, y_scores):
    puntos = sorted(zip(y_scores, y_true), reverse=True)
    tpr = []  # Tasa de verdaderos positivos
    fpr = []  # Tasa de falsos positivos
    TP = FP = 0
    FN = sum(y_true)
    TN = len(y_true) - FN

    for score, label in puntos:
        if label == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        tpr.append(TP / (TP + FN) if TP + FN > 0 else 0)
        fpr.append(FP / (FP + TN) if FP + TN > 0 else 0)

    return fpr, tpr

fpr, tpr = calcular_curva_roc(y_true, y_scores)

# Graficar la curva ROC
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('#e6f2ff')
ax = plt.gca()  # obtiene el eje actual
ax.set_facecolor('black')  # o cualquier color: 'white', '#f0f0f0', 'black', etc.
plt.plot(fpr, tpr, color='#FF0B55', lw=2, label='Curva ROC')
plt.plot([0, 1], [0, 1], color='olive', linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Naive Bayes (40-45 años)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white',
# 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'navy', 'teal'