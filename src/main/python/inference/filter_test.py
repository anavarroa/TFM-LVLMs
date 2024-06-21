import os
import json
import random
from collections import defaultdict

random.seed(28)

def dividir_dataset(data, ratio):
    indices = list(range(len(data)))
    random.shuffle(indices)
    selected_idx = int(len(indices) * ratio)

    indices_seleccionados = indices[:selected_idx]

    datos_seleccionados = [data[i] for i in sorted(indices_seleccionados)]

    return datos_seleccionados

def particionar_datos_por_carpetas(data, ratio):
    datos_por_carpeta = defaultdict(list)

    for entry in data:
        ruta_imagen = entry["image"]
        carpeta_principal = ruta_imagen.split('/')[0]
        datos_por_carpeta[carpeta_principal].append(entry)

    datos_seleccionados = []

    for carpeta, items in datos_por_carpeta.items():
        resultado = dividir_dataset(items, ratio)
        datos_seleccionados.extend(resultado)

    return datos_seleccionados

def guardar_json(data, ruta_archivo):
    with open(ruta_archivo, "w") as f:
        json.dump(data, f, indent=4)

def crear_dataset_reducido(directorio, ratio):
    # Ruta del archivo JSON original
    ruta_json = "/datassd/proyectos/tfm-alvaro/data/sets/data_test.json"

    # Cargar datos del JSON
    with open(ruta_json, "r") as f:
        data = json.load(f)

    # Dividir datos por carpetas y seleccionar el ratio especificado
    datos_seleccionados = particionar_datos_por_carpetas(data, ratio)

    # Guardar el conjunto reducido en un archivo JSON
    ruta_archivo_reducido = "/datassd/proyectos/tfm-alvaro/data/sets/filtered_test.json"
    guardar_json(datos_seleccionados, ruta_archivo_reducido)

    # Mensaje de confirmación
    print(f"Conjunto reducido creado. ({int(ratio*100)}%)")

if __name__=="__main__":
    directorio = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    # Ratio para el conjunto reducido
    ratio = 0.3 # 30%
    crear_dataset_reducido(directorio, ratio)
