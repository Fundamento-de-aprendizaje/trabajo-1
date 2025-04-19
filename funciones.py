import random

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
            #print(example)
      # else: 
            #print("NO CUMPLE", example)
    return contador

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
    print("Es la proporción de instancias que han sido correctamente clasificadas.\n")

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

def separarDatosEnConjuntos(datos,porcentaje):
    random.shuffle(datos)
    # Calcula el índice de división ("porcentaje" de los datos)
    tamano_entrenamiento = int(len(datos) * porcentaje)

    entrenamiento = datos[:tamano_entrenamiento]
    prueba = datos[tamano_entrenamiento:]   
    return entrenamiento, prueba