import os
import json
import random
from collections import defaultdict

random.seed(28)

def dividir_dataset(data, train_ratio, test_ratio):
    indices = list(range(len(data)))
    random.shuffle(indices)
    train_idx = int(len(indices) * train_ratio)
    test_idx = train_idx + int(len(indices) * test_ratio)

    indices_entrenamiento = indices[:train_idx]
    indices_prueba = indices[train_idx:test_idx]

    datos_entrenamiento = [data[i] for i in sorted(indices_entrenamiento)]
    datos_prueba = [data[i] for i in sorted(indices_prueba)]

    if test_idx < len(indices):
        indices_validacion = indices[test_idx:]
        datos_validacion = [data[i] for i in sorted(indices_validacion)]
        return datos_entrenamiento, datos_prueba, datos_validacion
    else:
        return datos_entrenamiento, datos_prueba, None

def particionar_datos_por_carpetas(data, train_ratio, test_ratio, crear_val):
    datos_por_carpeta = defaultdict(list)

    for entry in data:
        ruta_imagen = entry["image"]
        carpeta_principal = ruta_imagen.split('/')[0]
        datos_por_carpeta[carpeta_principal].append(entry)

    datos_entrenamiento = []
    datos_prueba = []
    datos_validacion = []

    for carpeta, items in datos_por_carpeta.items():
        resultado = dividir_dataset(items, train_ratio, test_ratio)
        datos_entrenamiento.extend(resultado[0])
        datos_prueba.extend(resultado[1])
        if crear_val and resultado[2] is not None:
            datos_validacion.extend(resultado[2])

    return datos_entrenamiento, datos_prueba, datos_validacion if datos_validacion else None

def guardar_json(data, ruta_archivo):
    with open(ruta_archivo, "w") as f:
        json.dump(data, f, indent=4)

def crear_datasets(directorio, train_ratio, test_ratio):
    # Ruta del archivo JSON original
    ruta_json = os.path.join(directorio, "dataset.json")

    # Cargar datos del JSON
    with open(ruta_json, "r") as f:
        data = json.load(f)

    # Determinar si se debe crear el conjunto de validación
    crear_val = (train_ratio + test_ratio) < 1

    # Dividir datos por carpetas y crear conjuntos de entrenamiento, prueba y validación
    datos_entrenamiento, datos_prueba, datos_validacion = particionar_datos_por_carpetas(data, train_ratio, test_ratio, crear_val)

    # Crear directorio 'sets' si no existe
    sets_dir = os.path.join(directorio, "sets")
    os.makedirs(sets_dir, exist_ok=True)

    # Guardar conjuntos de entrenamiento y prueba en archivos JSON
    guardar_json(datos_entrenamiento, os.path.join(sets_dir, "data_train.json"))
    guardar_json(datos_prueba, os.path.join(sets_dir, "data_test.json"))

    # Guardar conjunto de validación si existe
    if crear_val and datos_validacion:
        guardar_json(datos_validacion, os.path.join(sets_dir, "data_val.json"))

    # Mensaje de confirmación
    print(f"Conjunto de entrenamiento creado. ({int(train_ratio*100)}%)")
    print(f"Conjunto de prueba creado. ({int(test_ratio*100)}%)")
    if crear_val:
        print(f"Conjunto de validación creado. ({int((1-(train_ratio+test_ratio))*100)}%)")

if __name__=="__main__":
    directorio = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data")

    # Los ratios pueden cambiarse
    train_ratio = 0.7 #70%
    test_ratio = 0.2 #20%
    # val_ratio = 1-train_ratio-test_ratio
    crear_datasets(directorio, train_ratio, test_ratio)
