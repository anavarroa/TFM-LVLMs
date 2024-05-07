# TFM Modelos Multimodales



## Descarga de datos

El dataset de entrenamiento original se llama [**NWPU-RSICD-UAV-UCM-LR-DOTA-instructions**](https://huggingface.co/datasets/BigData-KSU/RS-instructions-dataset/blob/main/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json), y consta de datos de varios datasets de Remote Sensing. He editado dicho dataset para que contenga datos de únicamente cuatro conjuntos principales: NWPU, RSICD, LR y DOTA ([dataset.json](data/dataset.json)).

Deberán obtenerse las imágenes de dichos datasets (unas 45000) para poder empezar a trabajar .

- Las imágenes de **DOTA** y **LR** pueden descargarse automáticamente ejecutando el siguiente [script](src/main/python/script.py).
- Las imágenes del dataset **NWPU** deben descargarse manualmente desde el [OneDrive](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp).
- Las imágenes de **RSICD** deben descargarse manualmente desde el [Mega](https://mega.nz/folder/EOpjTAwL#LWdHVjKAJbd3NbLsCvzDGA).

Las imágenes descomprimidas deben alojarse en una carpeta "imagenes" dentro de "data". Para que todo funcione deben tener la misma estructura que indica en el archivo JSON del dataset (alojado en la misma carpeta):

```
data
├── imagenes
│   ├── RSICD
│   │   └── images
│   ├── NWPU
│   │   └── NWPU_images
│   ├── LR
│   │   └── Train
│   └── DOTA
│       └── images
└── dataset.json
```

Se recomienda ejecutar el [script](src/main/python/script.py) en primer lugar, pues creará las carpetas de manera correcta y descomprimirá las imágenes de LR y DOTA en el formato adecuado. El resto de datasets, NWPU y RSICD, deberán ser copiados con más cuidado; será necesario eliminar o renombrar las carpetas descomprimidas que no coincidan con la estructura mostrada. 