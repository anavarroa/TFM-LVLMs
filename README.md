# TFM Modelos Multimodales



## Descarga de datos

El dataset de entrenamiento original se llama [**NWPU-RSICD-UAV-UCM-LR-DOTA-instructions**](https://huggingface.co/datasets/BigData-KSU/RS-instructions-dataset/blob/main/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json), y consta de datos de varios datasets de _Remote Sensing_ (RS). El dataset deberá ser editado para que contenga datos de únicamente cuatro conjuntos principales: **NWPU**, **RSICD**, **LR** y **DOTA**. De ello se encargará el script [data_preparation](src/main/python/data_preparation.py) del que se habla abajo.

Deberán obtenerse las imágenes de dichos datasets (unas 45000) para poder empezar a trabajar .

- Las imágenes de **DOTA** y **LR** pueden descargarse automáticamente ejecutando el script [data_preparation](src/main/python/data_preparation.py).
- Las imágenes del dataset **NWPU** deben descargarse manualmente desde el [OneDrive](https://onedrive.live.com/?authkey=%21AHHNaHIlzp%5FIXjs&id=5C5E061130630A68%21107&cid=5C5E061130630A68&parId=root&parQt=sharedby&o=OneUp).
- Las imágenes de **RSICD** deben descargarse manualmente desde el [Mega](https://mega.nz/folder/EOpjTAwL#LWdHVjKAJbd3NbLsCvzDGA).

Las imágenes descomprimidas deben alojarse en una carpeta _data/imagenes_ paralela a las carpetas _src_ y _model_. Para que todo funcione deben tener la misma estructura que indica en el archivo _JSON_ del dataset (alojado en la misma carpeta):

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

Se recomienda ejecutar el script [data_preparation](src/main/python/script.py) en primer lugar, pues creará las carpetas de manera correcta, descomprimirá las imágenes de **LR** y **DOTA** en el formato adecuado y creará el archivo _JSON_ editado. Tardará unos cuantos minutos.

 El resto de datasets, **NWPU** y **RSICD**, deberán ser copiados con más cuidado; será necesario eliminar o renombrar las carpetas descomprimidas que no coincidan con la estructura mostrada.

 
## Descarga del modelo

El siguiente paso será descargar el modelo de [**LLaVA 1.5.**](https://github.com/haotian-liu/LLaVA.git). Para ello, en una nueva carpeta _model_ paralela a _src_ y _data_, deberemos clonar el repositorio de GitHub de **LLaVA 1.5.**:

```
git clone https://github.com/haotian-liu/LLaVA.git
```

una vez ejecutado, se creará una carpeta _LLaVA_ donde encontraremos todos los archivos del modelo. Para empezar a trabajar con el modelo deberemos estar situados dentro de esta carpeta

```
cd LLaVA
```

En esta carpeta ya podremos empezar a entrenar el modelo con nuestro conjunto de datos personalizado.

## Carga del modelo preentrenado

Vamos a cargar el modelo de LLaVA 1.5. preentrenado para empezar con el fine-tuning. El modelo puede encontrarse en [HuggingFace](https://huggingface.co/liuhaotian/llava-v1.5-7b), y viene con el tokenizer, el procesador de imágenes y el context length:

```
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    offload_folder="/content/llava_model"
)
```

### Rutas

Para empezar el proceso de fine-tuning, tenemos en primer lugar que indicar las rutas pertinentes que el modelo necesitará conocer:

```
DEEPSPEED_SCRIPT = "deepspeed llava/train/train_mem.py"
DEEPSPEED_JSON = "./scripts/zero3.json"
MODEL_NAME = "liuhaotian/llava-v.15-7b"
VISION_TOWER = "openai/clip-vit-large-patch14-336"
```

Deberemos también indicar la ruta a nuestro conjunto de datos personalizado, así como la carpeta donde deseamos que se alojen los resultados. Estas deberán ser especificadas manualmente:

- **DATA_PATH**: ruta al archivo _data/dataset.json_.
- **IMAGE_FOLDER**: ruta a la carpeta _data/imagenes_.
- **OUTPUT_DIR**: ruta a la carpeta donde queremos guardar los resultados.

En caso de ejecutar el código desde una carpeta dentro de _model/LLaVA_, y alojar los resultados en una carpeta _res_ paralela a _data_, _model_ y _src_, puede usarse el siguiente código:

```
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","data","dataset.json") # Cambiar según el caso
IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","data","imagenes") # Cambiar según el caso
OUTPUT_DIR = OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","res") # Cambiar según el caso
```

### Parámetros

Además de las rutas, es crucial configurar los parámetros del modelo.

```
finetune_script = f'''
{DEEPSPEED_SCRIPT} \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed {DEEPSPEED_JSON} \
    --model_name_or_path {MODEL_NAME} \
    --version v1 \
    --data_path {DATA_PATH} \
    --image_folder {IMAGE_FOLDER} \
    --vision_tower {VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir {OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
'''
```
Son estos los parámetros que iremos modificando para evaluar resultados.

- *lora_enable*: activa el uso de __LoRA__, una técnica de fine-tuning más eficiente basada en la utilización de matrices de menor rango.
- *lora_r*: rango de la descomposición de matrices en __LoRA__. El rango común en fine-tuning de LLMs es de 8 a 64, pero un mayor rango mejora la capacidad del modelo.
- *lora_alpha*: ayuda a preservar la estabilidad numérica. El valor habitual ronda 16.
- *mm_projector_lr*: ratio de aprendizaje separado para el proyector multimodal.
- *deepspeed*: especifica la configuración deepspeed zero stage 3 para el entrenamiento.
- *mm_projector_type*: fijado a un MLP con función de activación GeLu.
- *report_to wandb*: opción de monitoreo que proporciona un seguimiento del progreso y métricas de rendimiento a tiempo real.

El script utiliza **DeepSpeed**, una librería de optimización de PyTorch para Deep Learning diseñada para reducir el poder computacional y memoria a la hora de entrenar modelos en paralelo.
