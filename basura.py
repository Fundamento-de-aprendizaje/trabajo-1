# def predecir_naive_bayes(fila, conteo_clases, conteo_atributos, total_datos):
#     probabilidades = {}
#     #fila : una nueva observación sin la clase (ej: ['Edad', 'Sexo', '?'])
#     #conteo_clases  : dict con el total de ejemplos por clase
#     #conteo_atributos : dict con los conteos de atributos por clase
#     #total_datos : total de filas en el conjunto de entrenamiento

#     #calcula la probabilidad de que la fila pertenezca a cada clase y devuelve 
#     # la clase más probable (y también todas las probabilidades si querés verlas).
#     for clase in conteo_clases:
#         probabilidad_clase = conteo_clases[clase] / total_datos
#         probabilidad_atributos = probabilidad_clase

#         for i, atributo in enumerate(fila[:-1]):
#             if atributo in conteo_atributos[clase][i]:
#                 probabilidad_atributos *= conteo_atributos[clase][i][atributo] / conteo_clases[clase]
#                # print("\nif fila",conteo_atributos[clase][i],"valor ",conteo_atributos[clase][i][atributo])
#             else:
#                 probabilidad_atributos *= 0.0001  # Suavizado para evitar probabilidad 0
#                # print("\nelse fila",conteo_atributos[clase][i])

#         probabilidades[clase] = probabilidad_atributos
   
#     return max(probabilidades, key=probabilidades.get), probabilidades



 # fila : nueva observación (sin clase), ej: ['Adulto', 'M']
    # conteo_clases : dict con total de ejemplos por clase, ej: {'Si': 5, 'No': 3}
    # conteo_atributos : dict con conteos de atributos por clase, estructura:
    #    {'Si': {0: {'Joven': 2, 'Adulto': 3}, 1: {'M': 1, 'F': 4}}, ...}
    # total_datos : total de filas de entrenamiento
    # valores_posibles : dict que indica cuántos valores posibles tiene cada atributo:
    #    ej: {0: ['Joven', 'Adulto', 'Mayor'], 1: ['M', 'F']}
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