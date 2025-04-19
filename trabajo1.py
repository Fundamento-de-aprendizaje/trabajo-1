# @title
# Se debe tener instalado las siguientes librerias 
# pip install numpy pandas scikit-learn matplotlib

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
# hipotesisTrucha2= [35, '?', '?', '?', 'NO']
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


## FUNCIONES PARA AMBOS EJERCICIOS 
def metricas (TP,FP,TN,FN):
    print("\nMATRIZ DE CONFUSION")
    print(f"TP (TRUE POSITIVE): {TP}")
    print(f"FP (FALSE POSITIVE): {FP}")
    print(f"TN (TRUE NEGATIVE): {TN}")
    print(f"FN (FALSE NEGATIVE): {FN}")
    print(" ",TP,  "|", FN , "\n ",FP ,"|" ,TN )

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
    
    print(f"Accuracy(proporción correctamente clasificadas): {accuracy:.2f}")
    print("Es la proporci ́on de instancias que han sido correctamente clasificadas.\n")


    print(f"Recall (Sensibilidad): {recall:.2f}")
    print("Mide la probabilidad de que el clasificador detecte un caso Positivo cuando en verdad lo es.\n")


    print(f"Especificidad: {especificidad:.2f}")
    print(" Mide la probabilidad de que el clasificador detecte un caso Negativo cuando en verdad lo es.\n")


    print(f"Precisión: {precision:.2f}")
    print(" Mide la probabilidad de que el clasificador detecte correctamente un caso positivo..\n")


    print(f"F1-score: {f1_score:.2f}")
    print("  Combina las medidas de precision y recall para devolver una medida de calidad m ́as general del modelo.\n")

    print(f"Tasa de verdaderos positivos (TPR): {recall:.2f}")
    print(f"Tasa de falsos positivos (FPR): {fpr:.2f}")

# # Mostrar la matriz de confusión
# print("\n--- MATRIZ DE CONFUSIÓN ---")
# print(f"TP (TRUE POSITIVE): {TP}")
# print(f"FP (FALSE POSITIVE): {FP}")
# print(f"TN (TRUE NEGATIVE): {TN}")
# print(f"FN (FALSE NEGATIVE): {FN}")
# print(" ",TP,  "|", FN , "\n ",FP ,"|" ,TN )

# # Cálculos de métricas
# total = TP + FP + TN + FN
# accuracy = (TP + TN) / total
# recall = TP / (TP + FN) if TP + FN > 0 else 0
# especificidad = TN / (TN + FP) if TN + FP > 0 else 0
# precision = TP / (TP + FP) if TP + FP > 0 else 0
# f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
# fpr = FP / (FP + TN) if FP + TN > 0 else 0

# # Mostrar métricas
# print("\n--- MÉTRICAS ---")
 
# print(f"Accuracy(proporción correctamente clasificadas): {accuracy:.2f}")
# print("Es la proporci ́on de instancias que han sido correctamente clasificadas.\n")


# print(f"Recall (Sensibilidad): {recall:.2f}")
# print("Mide la probabilidad de que el clasificador detecte un caso Positivo cuando en verdad lo es.\n")


# print(f"Especificidad: {especificidad:.2f}")
# print(" Mide la probabilidad de que el clasificador detecte un caso Negativo cuando en verdad lo es.\n")

# # 
# print(f"Precisión: {precision:.2f}")
# print(" Mide la probabilidad de que el clasificador detecte correctamente un caso positivo..\n")


# print(f"F1-score: {f1_score:.2f}")
# print("  Combina las medidas de precision y recall para devolver una medida de calidad m ́as general del modelo.\n")

# print(f"Tasa de verdaderos positivos (TPR): {recall:.2f}")
# print(f"Tasa de falsos positivos (FPR): {fpr:.2f}")
metricas(TP,FP,TN,FN)
print(hypothesis)

###################################################
print("PUNTO 3")
print("PUNTO 3")
# Paso 1: Filtrar personas entre 40 y 45 años
personas_40_45 = datosPrestamos[(datosPrestamos['Edad'] >= 40) & (datosPrestamos['Edad'] <= 45)].copy()

# Convertir todas las columnas a tipo string para convertirlo en categórico
for col in personas_40_45.columns:
    personas_40_45[col] = personas_40_45[col].astype(str)

# Dividir aleatoriamente los datos en entrenamiento (80%) y prueba (20%)
datos = personas_40_45.values.tolist()
random.shuffle(datos)
tamano_entrenamiento = int(len(datos) * 0.8)
entrenamiento = datos[:tamano_entrenamiento]
prueba = datos[tamano_entrenamiento:]

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

# # Paso 4: Calcular métricas
# accuracy = (TP + TN) / (TP + FP + TN + FN)
# precision = TP / (TP + FP) if TP + FP > 0 else 0
# recall = TP / (TP + FN) if TP + FN > 0 else 0
# f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# # Mostrar resultados
# print("\n--- MATRIZ DE CONFUSIÓN 1 ---")
# print(f"TP (True Positive): {TP}")
# print(f"FN (False Negative): {FN}")
# print(f"FP (False Positive): {FP}")
# print(f"TN (True Negative): {TN}")
# print(" ", TP, "|", FN, "\n ", FP, "|", TN)

# print("\n--- MÉTRICAS ---")
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-score: {f1_score:.2f}")

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

