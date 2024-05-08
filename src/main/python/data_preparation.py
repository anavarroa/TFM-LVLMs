import os
import requests
import zipfile
import gdown
import json

def count_files(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg','.png','.tif')):
                count += 1
    return count


def descarga_LR(directorio):
    # URL del archivo zip a descargar
    url = "https://zenodo.org/record/6344334/files/Images_LR.zip"

    carpeta_lr = os.path.join(directorio, "LR")
    ruta_archivo_zip = os.path.join(directorio, "Images_LR.zip")
    ruta_archivo_extraido = os.path.join(carpeta_lr, "Train")

    # Crear la carpeta IMAGENES si no existe
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    # Crear la carpeta LR si no existe
    if not os.path.exists(carpeta_lr):
        os.makedirs(carpeta_lr)

    # Descargar el archivo zip
    respuesta = requests.get(url)
    with open(ruta_archivo_zip, "wb") as archivo_zip:
        archivo_zip.write(respuesta.content)

    # Descomprimir el archivo zip
    with zipfile.ZipFile(ruta_archivo_zip, 'r') as zip_ref:
        zip_ref.extractall(carpeta_lr)

    # Eliminar el archivo zip
    os.remove(ruta_archivo_zip)

    # Renombrar la carpeta extraida a "Train"
    os.rename(os.path.join(carpeta_lr, "Images_LR"), ruta_archivo_extraido)

    file_count = count_files(carpeta_lr)
    print("Dataset LR descargado:",file_count,"imágenes extraídas")




def descarga_DOTA(directorio):

    urls = [
        "https://drive.google.com/uc?id=1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2",  # part1.zip
        "https://drive.google.com/uc?id=1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v",  # part2.zip
        "https://drive.google.com/uc?id=1pEmwJtugIWhiwgBqOtplNUtTG2T454zn"   # part3.zip
    ]

    # Ruta donde guardar los archivos zip descargados
    carpeta_dota = os.path.join(directorio, "DOTA")

    # Crear la carpeta DOTA si no existe
    if not os.path.exists(carpeta_dota):
        os.makedirs(carpeta_dota)

    # Descargar y descomprimir los archivos zip
    for url in urls:
        # Obtener el ID del archivo desde la URL
        file_id = url.split("=")[-1]
        
        # Nombre del archivo zip
        zip_file_name = os.path.join(directorio, f"{file_id}.zip")
        
        # Descargar el archivo zip
        gdown.download(url, zip_file_name, quiet=False)

        with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
            zip_ref.extractall(carpeta_dota)
        
        # Eliminar el archivo zip
        os.remove(zip_file_name)

    file_count = count_files(carpeta_dota)
    print("Dataset DOTA descargado:",file_count,"imágenes extraídas")


def descarga_JSON(directorio):

    # URL del archivo JSON
    url = "https://huggingface.co/datasets/BigData-KSU/RS-instructions-dataset/resolve/main/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json"

    # Nombre del archivo destino
    filename = "dataset.json"

    # Descargar el archivo JSON
    response = requests.get(url)
    data = response.json()

    # Eliminar datos correspondientes a UAV
    filtered_data = [entry for entry in data if not entry["image"].startswith("UAV/Train")]

    # Eliminar los dos últimos bloques del json ("ocr_vqa")
    filtered_data = filtered_data[:-2]

    # Guardar el resultado en un nuevo archivo JSON con indentación
    output_path = os.path.join(directorio, filename)
    with open(output_path, "w") as f:
        json.dump(filtered_data, f, indent=4)

    print("archivo JSON descargado")




if __name__=='__main__':

    carpeta_imagenes = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data","imagenes")
    carpeta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data")

    descarga_LR(carpeta_imagenes)
    descarga_DOTA(carpeta_imagenes)
    descarga_JSON(carpeta_json)

    print('Proceso terminado!')