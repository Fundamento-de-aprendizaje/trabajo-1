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

def aplicarHipotesis(datosDePrueba, hypothesis):
    prediccion = 0

    formato = "[ {:<5} | {:<10} | {:<15} | {:<10} | {:<10} | {:<10} | {:<10} ]"
    print(formato.format("Edad", "Sexo", "Educación", "Vivienda", "Préstamos", "Real", "Predicción"))

    for dato in datosDePrueba:
        cumple = True
        for i in range(len(hypothesis)):
            if hypothesis[i] != '?' and hypothesis[i] != dato[i]:
                cumple = False
                break

        pred = "RECHAZADO"
        if cumple:
            prediccion += 1
            pred = "OTORGADO"

        print(formato.format(dato[0], dato[1], dato[2], dato[3], dato[4], dato[5], pred))

    return prediccion

def calcularMatrizDeConfusion(dataSet,hypothesis=[] ):
    TP = FP = TN = FN = 0

    for persona in dataSet:
        
        estadoPrestamo = persona[-1]  # Última columna es la etiqueta real
        prediccion = "OTORGADO"
        for i in range(len(hypothesis)):
            if hypothesis[i] != "?" and hypothesis[i] != persona[i]:
                prediccion = "RECHAZADO"
                break

        if prediccion == "OTORGADO":
            if estadoPrestamo == "OTORGADO":
                TP += 1
            else:
                FP += 1
        else:
            if estadoPrestamo == "RECHAZADO":
                TN += 1
            else:
                FN += 1

            FN += 1  
            
    print("\nMATRIZ DE CONFUSION PUNTO 2\n")
    print(" ",TP,  "|", FN , "\n ",FP ,"|" ,TN )   
    print(f"\nTP (TRUE POSITIVE): {TP}")
    print(f"FP (FALSE POSITIVE): {FP}")
    print(f"TN (TRUE NEGATIVE): {TN}")
    print(f"FN (FALSE NEGATIVE): {FN}\n")            
      
    

    return TP, FP, TN, FN


def calcularMetricas (TP,FP,TN,FN):

    # Cálculos de métricas
    total = TP + FP + TN + FN # esta bien
    accuracy = (TP + TN) / total
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    especificidad = TN / (TN + FP) if TN + FP > 0 else 0
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    fpr = FP / (FP + TN) if FP + TN > 0 else 0

    # Mostrar métricas
    print("\n--- MÉTRICAS ---")
    
    print(f"ACCURACY(proporción correctamente clasificadas): {accuracy:.2f}\n")
    

    print(f"RECALL (Sensibilidad): {recall:.2f}")
    print("Recall: mide la probabilidad de que el clasificador detecte un caso Positivo cuando en verdad lo es.\n")

    print(f"ESPECIFICIDAD: {especificidad:.2f}")
    print("Especificidad: Mide la probabilidad de que el clasificador detecte un caso Negativo cuando en verdad lo es.\n")

    print(f"PRECISIÓN: {precision:.2f}")
    print("Precisión: Mide la probabilidad de que el clasificador detecte correctamente un caso positivo..\n")

    print(f"F1-SCORE: {f1_score:.2f}")
    print("F1-score: Combina las medidas de precision y recall para devolver una medida de calidad m ́as general del modelo.\n")

    print(f"Tasa de verdaderos positivos (TPR): {recall:.2f}")
    print(f"Tasa de falsos positivos (FPR): {fpr:.2f}")

def separarDatosEnConjuntos(datos,porcentaje,mezclar):
    if(mezclar):
        random.shuffle(datos)
        
    # Calcula el índice de división ("porcentaje" de los datos)
    tamano_entrenamiento = int(len(datos) * porcentaje)

    entrenamiento = datos[:tamano_entrenamiento]
    prueba = datos[tamano_entrenamiento:]   
    return entrenamiento, prueba


def entrenar_naive_bayes(datos_entrenamiento):
    conteo_clases = {}
    conteo_atributos = {}

    for persona in datos_entrenamiento:
        clase = persona[-1]
        if clase not in conteo_clases:
            conteo_clases[clase] = 0   #inicializa cada uno de los conjuntos de atributos
            conteo_atributos[clase] = [{} for _ in range(len(persona) - 1)]
        conteo_clases[clase] += 1

        for i, atributo in enumerate(persona[:-1]): #recorre cada fila 
            if atributo not in conteo_atributos[clase][i]:# inicializa el conteo del atributo si corresponde
                conteo_atributos[clase][i][atributo] = 0
            conteo_atributos[clase][i][atributo] += 1 # cuenta la cantidad de veces que aparece dicho atributo
    #print("conteo_clases: ",conteo_clases, "\nconteo_atributos: ", conteo_atributos)
    return conteo_clases, conteo_atributos



def predecir_naive_bayes(persona, conteo_clases, conteo_atributos, datos_de_prueba, valores_posibles):
    probabilidades = {}
    suavizado=1   
    k= len(conteo_clases)  
   
    for clase in conteo_clases:
       
        # respeta completamente la fórmula con Laplace,suma 1 al numerador y suma K al denominador.
        probabilidad_clase = (conteo_clases[clase] + suavizado) / (datos_de_prueba + k)# La probabilidad a priori de la clase  tiene suavizado.
        probabilidad_total = probabilidad_clase                  # k en el denominador
        
        for i, atributo in enumerate(persona[:-1]):
            posibles_valores = len(valores_posibles[i]) 
            conteo_valor = conteo_atributos[clase][i].get(atributo, 0)
            #Las probabilidades condicionales (atributo dado clase) están bien con suavizado.
            probabilidad_atributo_dado_clase = (conteo_valor + suavizado) / (conteo_clases[clase] + suavizado*posibles_valores)
            probabilidad_total *= probabilidad_atributo_dado_clase

        probabilidades[clase] = probabilidad_total

    # Devolvemos la clase más probable y también todas las probabilidades
    return max(probabilidades, key=probabilidades.get), probabilidades

def calcular_curva_roc(y_true, y_scores):
    puntos = sorted(zip(y_scores, y_true), reverse=True)
    tpr = []  # Tasa de verdaderos positivos
    fpr = []  # Tasa de falsos positivos
    umbrales = []
    TP = FP = 0
    FN = sum(y_true)
    TN = len(y_true) - FN

    # print("Puntosssss", puntos)

    for score, label in puntos:
        if label == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1
            TN -= 1
        tpr.append(TP / (TP + FN) if TP + FN > 0 else 0)
        fpr.append(FP / (FP + TN) if FP + TN > 0 else 0)
        umbrales.append(score)
        print(score)


    return fpr, tpr,umbrales


    