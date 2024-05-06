
import os
import requests
import zipfile
import gdown

# URL del archivo zip a descargar
url = "https://zenodo.org/record/6344334/files/Images_LR.zip"

# Ruta donde guardar el archivo zip descargado
carpeta_imagenes = os.path.join("data", "imagenes")

carpeta_lr = os.path.join(carpeta_imagenes, "LR")
ruta_archivo_zip = os.path.join(carpeta_imagenes, "Images_LR.zip")
ruta_archivo_extraido = os.path.join(carpeta_lr, "Train")

# Crear la carpeta IMAGENES si no existe
if not os.path.exists(carpeta_imagenes):
    os.makedirs(carpeta_imagenes)

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

print("LR completado")


urls = [
    "https://drive.google.com/uc?id=1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2",  # part1.zip
    "https://drive.google.com/uc?id=1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v",  # part2.zip
    "https://drive.google.com/uc?id=1pEmwJtugIWhiwgBqOtplNUtTG2T454zn"   # part3.zip
]

# Ruta donde guardar los archivos zip descargados
carpeta_dota = os.path.join(carpeta_imagenes, "DOTA")

# Crear la carpeta DOTA si no existe
if not os.path.exists(carpeta_dota):
    os.makedirs(carpeta_dota)

# Descargar y descomprimir los archivos zip
for url in urls:
    # Obtener el ID del archivo desde la URL
    file_id = url.split("=")[-1]
    
    # Nombre del archivo zip
    zip_file_name = os.path.join(carpeta_imagenes, f"{file_id}.zip")
    
    # Descargar el archivo zip
    gdown.download(url, zip_file_name, quiet=False)

    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall(carpeta_dota)
    
    # Eliminar el archivo zip
    os.remove(zip_file_name)

print("DOTA completado")
