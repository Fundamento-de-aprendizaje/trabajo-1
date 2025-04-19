# @title
# Se debe tener instalado las siguientes librerias 
# pip install numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
import random
from funciones import find_s, aplicarHipotesis, evaluar_hipotesis, metricas, separarDatosEnConjuntos

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
print(datosPrestamos.columns.to_list())
# Mostrar la hipótesis final
print( "  ",hypothesis[0],"        ",hypothesis[1],"               ",hypothesis[2],"                ",hypothesis[3],"                ",hypothesis[4])
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
def entrenar_naive_bayes(datos_entrenamiento):
    conteo_clases = {}
    conteo_atributos = {}

    for fila in datos_entrenamiento:
        clase = fila[-1]
        if clase not in conteo_clases:
            conteo_clases[clase] = 0   #inicializa cada uno de los conjuntos de atributos
            conteo_atributos[clase] = [{} for _ in range(len(fila) - 1)]
        conteo_clases[clase] += 1

        for i, atributo in enumerate(fila[:-1]):
            if atributo not in conteo_atributos[clase][i]:
                conteo_atributos[clase][i][atributo] = 0
            conteo_atributos[clase][i][atributo] += 1
    print("conteo_clases: ",conteo_clases, "\nconteo_atributos: ", conteo_atributos)
    return conteo_clases, conteo_atributos
   
def predecir_naive_bayes(fila, conteo_clases, conteo_atributos, total_datos):
    probabilidades = {}
    for clase in conteo_clases:
        probabilidad_clase = conteo_clases[clase] / total_datos
        probabilidad_atributos = probabilidad_clase

        for i, atributo in enumerate(fila[:-1]):
            if atributo in conteo_atributos[clase][i]:
                probabilidad_atributos *= conteo_atributos[clase][i][atributo] / conteo_clases[clase]
               # print("\nif fila",conteo_atributos[clase][i],"valor ",conteo_atributos[clase][i][atributo])
            else:
                probabilidad_atributos *= 0.0001  # Suavizado para evitar probabilidad 0
               # print("\nelse fila",conteo_atributos[clase][i])

        probabilidades[clase] = probabilidad_atributos
   
    return max(probabilidades, key=probabilidades.get), probabilidades

# Entrenar el modelo
conteo_clases, conteo_atributos = entrenar_naive_bayes(entrenamiento)

# Paso 3: Evaluar el modelo en el conjunto de prueba
TP = FP = TN = FN = 0
y_true = []
y_scores = []



for fila in prueba:
    prediccion, probabilidades = predecir_naive_bayes(fila, conteo_clases, conteo_atributos, len(entrenamiento))
    verdadero_estado = fila[-1]
    y_true.append(1 if verdadero_estado == "OTORGADO" else 0)
    y_scores.append(probabilidades["OTORGADO"] if "OTORGADO" in probabilidades else 0)

    if prediccion == "OTORGADO":
        if verdadero_estado == "OTORGADO":
            TP += 1
        else:
            FP += 1
    else:
        if verdadero_estado == "RECHAZADO":
            TN += 1
        else:
            FN += 1


# USAMOS UNA FUNCION PARA EL PUNTO 2 y 3. Para la parte de metricuas.

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

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC - Naive Bayes (40-45 años)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

