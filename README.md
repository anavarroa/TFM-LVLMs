# TFM Modelos Multimodales



## Descarga de datos

El dataset que se usará para entrenar el modelo se llama NWPU-RSICD-UAV-UCM-LR-DOTA-instructions, y consta de datos de varios datasets de Remote Sensing. He editado dicho dataset para que contenga datos de únicamente cuatro conjuntos principales: NWPU, RSICD, LR y DOTA.

Deberán obtenerse las imágenes de dichos datasets para poder empezar a trabajar (unas 45000).

- Las imágenes de **DOTA** y **LR** pueden descargarse automáticamente ejecutando el siguiente [script](src/main/python/script.py).
- Las imágenes del dataset **NWPU** deben descargarse manualmente desde el [OneDrive](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp).
- Las imágenes de **RSICD** deben descargarse manualmente desde el [Mega](https://mega.nz/folder/EOpjTAwL#LWdHVjKAJbd3NbLsCvzDGA).

Las imágenes deben alojarse en una carpeta "data" paralela a "src". Para que todo funcione deben tener la misma estructura que indica en el dataset, es decir:

```
data
└── imagenes
    ├── RSICD
    │   └── images
    ├── NWPU
    │   └── NWPU_images
    ├── LR
    │   └── Train
    └── DOTA
        └── images
```

Habrá por tanto que eliminar o renombrar las carpetas descomprimidas de RSICD y NWPU que no coincidan.