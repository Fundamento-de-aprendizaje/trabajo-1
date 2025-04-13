# @title
# Se debe tener instalado las siguientes librerias 
# pip install numpy
# pip install pandas
# pip install random

import numpy as np
import pandas as pd
import random

# leemos los datos del csv y le indicamos que columnas vamos a utilizar
url = 'https://drive.google.com/uc?export=download&id=1BQEFonHa5aYO4MTg1EWxGIuRSgCw6ZXb'
# leemos los datos del csv y le indicamos que columnas vamos a utilizar
datosPrestamos = pd.read_csv(url, usecols=[0, 1, 2, 5,12,13], encoding='latin1')
#print(datosPrestamos.head())

def find_s(examples):
    # Inicializar la hipótesis con el primer ejemplo positivo
    hypothesis = examples[0][:len(examples[0])-1]  # Excluimos la etiqueta (última columna)

    # Iteramos sobre los ejemplos positivos
    for example in examples[1:]:
        if example[-1] == "OTORGADO":  # Solo consideramos ejemplos positivos
            # Comparamos el ejemplo con la hipótesis
            for i in range(len(hypothesis)):
                if hypothesis[i] != example[i] and hypothesis[i] != "?":
                    # Si no son iguales, ajustamos la hipótesis (más general)
                    hypothesis[i] = "?"

    return hypothesis

# Filtrar solo personas de 50 años
personasDe50Anios = datosPrestamos[datosPrestamos['Edad'] == 50].values.tolist()
personasRandom = personasDe50Anios.copy()
random.shuffle(personasRandom)
# Calcula el índice de división (75% de los datos)
tamano_entrenamiento = int(len(personasRandom) * 0.75)



entrenamiento = personasRandom[:tamano_entrenamiento]
prueba = personasRandom[tamano_entrenamiento:]
########################

# Llamar al algoritmo Find-S
hypothesis = find_s(entrenamiento)

# Mostrar la hipótesis final
print(datosPrestamos.columns.to_list())
print("La hipótesis final es:", hypothesis)


print("Personas con 50 años:   ",len(personasDe50Anios))
print("Entrenamiento:  ",len(entrenamiento))
print("Prueba:   ",len(prueba))

# def aplicarHipotesis(examples, hypothesis):
#     # Inicializar la hipótesis con el primer ejemplo positivo
#     contador = 0
#     # Iteramos sobre los ejemplos positivos
#     for example in examples[1:]:
#       #  if example[-1] == "OTORGADO":  # Solo consideramos ejemplos positivos
#             # Comparamos el ejemplo con la hipótesis
#             for i in range(len(hypothesis)):
#                 if hypothesis[i] == '?' or  hypothesis[i] == example[i]:
#                     # Si no son iguales, ajustamos la hipótesis (más general)
#                     contador = contador + 1
#                     print(example)

#     return contador
def aplicarHipotesis(examples, hypothesis):
    contador = 0
    for example in examples:
        cumple = True
        for i in range(len(hypothesis)):
            if hypothesis[i] != '?' and hypothesis[i] != example[i]:
                cumple = False
                break
        if cumple:
            contador += 1
            print(example)
        else: 
            print("NO CUMPLE", example)
    return contador
# hipotesisTrucha2= [50, '?', '?', 'INQUILINO', 'NO']
contador=   aplicarHipotesis(prueba,hypothesis)
print("\nPREDICCIÓN    ", contador,"  APROBADOS DE UN TOTAL DE ",len(prueba)  )
print(" \nEN PORCENTAJE    ", int(contador/len(prueba)*100) ," % "  )


############################################################
# PUNTO 2
def evaluar_hipotesis(hypothesis, datos_prueba):
    TP = FP = TN = FN = 0

    for ejemplo in datos_prueba:
        verdadero_estado = ejemplo[-1]  # Última columna es la etiqueta real
        prediccion = "OTORGADO"
        for i in range(len(hypothesis)):
            if hypothesis[i] != "?" and hypothesis[i] != ejemplo[i]:
                prediccion = "RECHAZADO"
                break

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

    return TP, FP, TN, FN

# Obtener los valores
TP, FP, TN, FN = evaluar_hipotesis(hypothesis, prueba)

# Mostrar la matriz de confusión
print("\n--- MATRIZ DE CONFUSIÓN ---")
print(f"TP (TRUE POSITIVE): {TP}")
print(f"FP (FALSE POSITIVE): {FP}")
print(f"TN (TRUE NEGATIVE): {TN}")
print(f"FN (FALSE NEGATIVE): {FN}")
print(" ",TP,  "|", FP, "\n ", FN ,"|" ,TN )

# Cálculos de métricas
total = TP + FP + TN + FN
accuracy = (TP + TN) / total
recall = TP / (TP + FN) if TP + FN > 0 else 0
especificidad = TN / (TN + FP) if TN + FP > 0 else 0
precision = TP / (TP + FP) if TP + FP > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
fpr = FP / (FP + TN) if FP + TN > 0 else 0

# Mostrar métricas
print("\n--- MÉTRICAS ---")
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall (Sensibilidad): {recall:.2f}")
print(f"Especificidad: {especificidad:.2f}")
print(f"Precisión: {precision:.2f}")
print(f"F1-score: {f1_score:.2f}")
print(f"Tasa de verdaderos positivos (TPR): {recall:.2f}")
print(f"Tasa de falsos positivos (FPR): {fpr:.2f}")
