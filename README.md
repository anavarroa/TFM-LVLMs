# LLaVA 1.5-7B para tareas de Image Captioning y VQA con un dataset de Remote Sensing personalizado.

<div align="center">
  <img src="imagen.jpeg" alt="LLaVA 1.5 para RS" width="25%">
</div>

## Conjunto de datos

El dataset de entrenamiento original se llama [**NWPU-RSICD-UAV-UCM-LR-DOTA-instructions**](https://huggingface.co/datasets/BigData-KSU/RS-instructions-dataset/blob/main/NWPU-RSICD-UAV-UCM-LR-DOTA-intrcutions.json), y consta de datos de varios datasets de _Remote Sensing_ (RS). El dataset deberá ser editado para que contenga datos de únicamente cuatro conjuntos principales: **NWPU**, **RSICD**, **LR** y **DOTA**. De ello se encargará el script [data_preparation](src/main/python/data_preparation.py) del que se habla abajo.

### Descarga del dataset

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

Se recomienda ejecutar el script [data_preparation](src/main/python/data_preparation.py) en primer lugar, pues creará las carpetas de manera correcta, descomprimirá las imágenes de **LR** y **DOTA** en el formato adecuado y creará el archivo _JSON_ editado. Tardará unos cuantos minutos.

 El resto de datasets, **NWPU** y **RSICD**, deberán ser copiados con más cuidado; será necesario eliminar o renombrar las carpetas descomprimidas que no coincidan con la estructura mostrada.


### Partición

Será necesario hacer una partición del conjunto de datos elegido. Esta partición debe ser la misma durante todo el proceso de entrenamiento y posterior evaluación:
- **Conjunto de entrenamiento** (*data_train*): datos que se usarán para entrenar al modelo, y que comprenderá la mayor parte del dataset.
- **Conjunto de validación** (*data_val*): datos que se usarán, durante el entrenamiento, para evaluar el rendimiento y ajustar los hiperparámetros.
- **Conjunto de prueba** (*data_test*): datos que se usarán para comprobar la calidad del modelo ya entrenado, durante la evaluación de resultados.

El script [data_partition](src/main/python/data_partition.py) llevará a cabo la partición mencionada del conjunto de datos, creando tres archivos JSON correspondientes a los tres conjuntos a considerar, y los ubicará en una carpeta _sets_ dentro de _data_:

```
data
├── imagenes
├── sets
│   ├── data_train.json
│   ├── data_val.json
│   └── data_test.json
└── dataset.json
```

El porcentaje de datos en cada conjunto es por defecto 70% _TRAIN_, 10% _VAL_ y 20% _TEST_ (con semilla fijada). Si se desea pueden modificarse estos porcentajes desde el script, en la función *crear_datasets(directorio, train_ratio, test_ratio)*:
- Si *train_ratio+test_ratio=1* se crearán el conjunto de entrenamiento y el de prueba
- Si *train_ratio+test_ratio<1* se crearán, además de dichos conjuntos, el conjunto de validación con el porcentaje restante de datos.

De esta forma la partición se llevará a cabo de forma automática según los intereses, si bien para nuestro caso consideraremos únicamente los conjuntos de entrenamiento y prueba.


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

Vamos a cargar el modelo de LLaVA 1.5. preentrenado para empezar con el fine-tuning. El modelo puede encontrarse en [HuggingFace](https://huggingface.co/liuhaotian/llava-v1.5-7b), y viene con el tokenizer, el procesador de imágenes y el context length.

### Rutas

Para empezar el proceso de fine-tuning, tenemos en primer lugar que indicar las rutas pertinentes que el modelo necesitará conocer:

```
DEEPSPEED_SCRIPT = "deepspeed model/LLaVA/llava/train/train_mem.py"
DEEPSPEED_JSON = "model/LLaVA/scripts/zero3.json"
MODEL_NAME = "liuhaotian/llava-v1.5-7b"
VISION_TOWER = "openai/clip-vit-large-patch14-336"
```

Deberemos también indicar la ruta a nuestro conjunto de datos personalizado, así como la carpeta donde deseamos que se alojen los resultados. Estas deberán ser especificadas manualmente:

- **TRAIN_DATA_PATH**: ruta al conjunto de entrenamiento *data/sets/data_train.json*.
- **VAL_DATA_PATH**: ruta al conjunto de validación *data/sets/data_val.json*.
- **IMAGE_FOLDER**: ruta a la carpeta _data/imagenes_.
- **OUTPUT_DIR**: ruta a la carpeta donde queremos guardar los resultados.

En caso de ejecutar el código desde el mismo directorio donde se aloja el script *data_preparation*, y alojar los resultados en una carpeta _res_ paralela a _data_, _model_ y _src_, puede usarse el siguiente código:

```
TRAIN_DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data","sets","data_train.json")) # Cambiar según el caso
IMAGE_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data","imagenes")) # Cambiar según el caso
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","res")) # Cambiar según el caso
```

NOTA: para poder usar el conjunto de validación durante el entrenamiento será necesario hacer unas cuantas modificaciones al script de entrenamiento _train.py_, pues está diseñado para considerar únicamente el conjunto de entrenamiento.

### Parámetros

Además de las rutas, es crucial configurar los parámetros del modelo. El siguiente stream ejecutará el entrenamiento con __LORA__, una técnica de fine-tuning más eficiente basada en la utilización de matrices de menor rango. Los parámetros ajustables son los que aparecen en él.

```
finetune_script = f'''
{DEEPSPEED_SCRIPT} \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed {DEEPSPEED_JSON} \
    --model_name_or_path {MODEL_NAME} \
    --version v1 \
    --training_data_path {TRAIN_DATA_PATH} \
    --validation_data_path {VAL_DATA_PATH} \
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
Algunos de los parámetros más importantes del stream se definen en las clases _ModelArguments_, _DataArguments_ y _TrainingArguments_ del script _train.py_, ubicado en _model/LLaVA/llava/train_.

Vamos ahora con una breve explicación del resto de parámetros que aparecen en el stream:

- *lora_enable*: activa el uso de __LoRA__.
- *lora_r*: rango de la descomposición de matrices en __LoRA__.
- *lora_alpha*: ayuda a preservar la estabilidad numérica. El valor habitual ronda 16.
- *mm_projector_lr*: tasa de aprendizaje separado para el proyector multimodal.
- *deepspeed*: especifica la configuración deepspeed zero stage 3 para el entrenamiento.
- *mm_projector_type*: tipo de proyector multimodal, fijado a un MLP con función de activación GeLu.
- *mm_vision_select_layer*: especifica la capa del modelo de visión que usará para la fusión multimodal.
- *mm_use_im_start_end*: indica si se utilizan tokens especiales para marcar el inicio y final de una imagen.
- *mm_use_im_patch_tokens*: indica si se usan tokens de parche de imagen.
- *group_by_modality_length*: controla cómo se agrupan los lotes de entreamiento basados en la longitud de las secuencias de diferentes modalidades.
- *num_train_epochs*: número de épocas de entrenamiento.
- *per_device_train_batch_size*: cantidad de ejemplos de entrenamiento que se utilizan en una sola iteración del algoritmo.
- *gradient_accumulation_steps*: acumulación de gradiente antes de actualizar los pesos.
- *evaluation_strategy*: indica si se realizará evaluación durante el entrenamiento.
- *learning_rate*: tasa de aprendizaje.
- *weight_decay*: regularización.
- *warmup_ratio*: programación de calentamiento.
- *logging_steps*: frecuencia con que se registran métricas durante el entrenamiento.
- *model_max_length*: longitud máxima para las secuencias de entrada.
- *gradient_checkpoint*: habilita el checkpoint para ahorrar memoria durante el entrenamiento.
- *dataloader_num_workers*: número de procesos de carga de datos.
- *report_to wandb*: opción de monitoreo que proporciona un seguimiento del progreso y métricas de rendimiento a tiempo real.

#### Parámetros Importantes

De todos los parámetros, es pertinente identificar cuáles van a ser los más importantes, pues serán los que habrá que modificar para evaluar los resultados obtenidos. Dichos parámetros son, principalmente
- lora_r: el rango común en fine-tuning de LLMs es de 8 a 64.
$$lora\_ r\in[8,64]$$
- num_train_epochs: el número de épocas suele ser de 1 a 10, si bien valores de 3 o 4 son bastante comunes.
$$num\_ train\_ epochs\in[1,10]$$
- gradient_accumulation_steps: generalmente se escoge un valor entre 1 y 10.
$$gradient\_ accumulation\_ steps\in[1,10]$$
- learning_rate: se suele tomar una tasa de aprendizaje en torno a los órdenes de e-5 hasta e-3, siendo comunes los órdenes de e-4.
$$learning\_ rate\in[2\cdot10^{-5},4\cdot10^{-3}]$$


En caso de lanzarlo sin LORA, deberá eliminarse toda la primera línea del stream:
```
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
```
Esto, no obstante, puede llegar a dar problemas por falta de memoria en GPU.

El stream anterior utiliza **DeepSpeed**, una librería de optimización de Deep Learning que proporciona una amplia gama de funciones para acelerar el entrenamiento de modelos grandes en GPU y sistemas distribuidos

## Entrenamiento
Antes de nada, es recomendable borrar el cache de **CUDA** para asegurar un uso de memoria eficiente:

```
import torch

torch.cuda.empty_cache()
```

A continuación, para ejecutar el stream podemos usar la librería _subprocess_:

```
import subprocess

result = subprocess.run([finetune_script], shell=True, capture_output=True, text=True)
print(result.stdout)
```

### Errores
Si da problemas, puede ser por varios motivos:
- Rutas mal especificadas o enlaces incorrectos.
- Librerías no instaladas como flash-attn ```pip install flash-attn --no-build-isolation```, indispensable para el funcionamiento del script _train.py_.
- No suficiente memoria en GPU: con cuatro **Nvidia H100 de 80GB** puede llegar a ocupar +50000Mib de memoria en cada una. Para lidiar con este problema puede disminuirse el *batch_size* (el mínimo es 2).
- Error de subprocess: en este caso se puede probar a hacer un print del stream ```print(finetune_script)``` y ejecutarlo **desde terminal**. De esta forma al menos se puede identificar el error más fácilmente, pues el mensaje dado por _subprocess_ no es muy concreto.

### Éxito

Si todo va bien, antes de comenzar el entrenamiento aparecerá el siguiente mensaje de _wandb_:
```
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 
```
Puede crearse una cuenta para poder llevar un seguimiento en directo de todo el proceso de entrenamiento desde la página [**Weights & Biases**](https://wandb.ai/site) (en tal caso se requerirá una ID que habrá que introducir por terminal). De lo contrario, puede elegirse no visualizar el resultado y proseguir con el entrenamiento. **Wandb** es una buena herramienta para realizar un estudio en detalle, pero si no se piensa usar puede eliminarse la última orden del stream, ```report_to wandb```. El cuadro de diálogo que crea _Wandb_ dará problemas si el programa es ejecutado en segundo plano. Sin embargo, una vez se haya lanzado una vez se registrará la ID de **W&B**, y podrá ejecutarse sin problemas con _nohup_:

```
nohup deepspeed model/LLaVA/llava/train/train_mem.py     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5     --deepspeed model/LLaVA/scripts/zero3.json     --model_name_or_path liuhaotian/llava-v1.5-7b     --version v1     --data_path /datassd/home/anavarroa/tfm-modelos-multimodales/data/dataset.json     --image_folder /datassd/home/anavarroa/tfm-modelos-multimodales/data/imagenes     --vision_tower openai/clip-vit-large-patch14-336     --mm_projector_type mlp2x_gelu     --mm_vision_select_layer -2     --mm_use_im_start_end False     --mm_use_im_patch_token False     --image_aspect_ratio pad     --group_by_modality_length True     --bf16 True     --output_dir /datassd/home/anavarroa/tfm-modelos-multimodales/res     --num_train_epochs 0.05     --per_device_train_batch_size 16     --per_device_eval_batch_size 4     --gradient_accumulation_steps 1     --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 50000     --save_total_limit 1     --learning_rate 2e-4     --weight_decay 0.     --warmup_ratio 0.03     --lr_scheduler_type "cosine"     --logging_steps 1     --tf32 True     --model_max_length 2048     --gradient_checkpointing True     --dataloader_num_workers 4     --lazy_preprocess True     --report_to wandb > log.txt &
```

Obtener un error pasado este punto es muy posiblemente debido a una falta de memoria en GPU (reducir *batch_size* en tal caso). En otra terminal puede estudiarse el uso de memoria del entrenamiento a tiempo de ejecución real (actualizado cada medio segundo) mediante:
```
watch -n 0.5 nvidia-smi
```

Llegado el momento una barra de carga irá completándose hasta suplir el *num_train_epochs* indicadas, convergiendo al *learning_rate* e indicando la pérdida (*loss*). Esto puede tomar un tiempo largo, dependiendo del número de épocas. El proceso terminará con un historial y resumen de ejecución, guardado en el archivo _log.txt_, el cual conviene guardar en algún lugar. Al terminar el proceso se habrá creado una carpeta _wandb_ paralela a _data_, _model_ y _src_ con todos los datos del proceso, así como una serie de resultados en varios formatos dentro de la carpeta indicada por **OUTPUT_DIR**.

```
root
├── data
├── model
├── src
├── res
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── config.json
│   ├── non_lora_trainables.bin
│   ├── README.md
│   └── trainer_state.json
├── logs
│   └── log.txt
└── wandb
    └── ...
```

- *adapter_config.json*: contiene la configuración específica del adaptador **LoRA**, e incluye detalles sobre cómo se integran los adaptadores en el modelo base.
- *adapter_model.safetensors*:  almacena los pesos del adaptador **LoRA**.
- *config.json*: contiene la configuración del modelo, incluyendo detalles como la arquitectura del modelo, el vocabulario y otros parámetros.
- *non_lora_trainables.bin*: almacena los pesos entrenados de las partes del modelo que no están cubiertas por los adaptadores **LoRA**.
- *trainer_state.json*: contiene el estado del entrenador, incluyendo el estado del optimizador, el scheduler de aprendizaje, y otras estadísticas del entrenamiento.

La carpeta _wandb_ almacena los registros y artefactos de entrenamiento enviados a **W&B**. Incluye gráficos de métricas de entrenamiento y otros datos de seguimiento que pueden consultarse en la página de **Weights & Biases**, como el la pérdida, la época y la tasa de aprendizaje en función del paso del entrenamiento.

Los checkpoints del entrenamiento (marcados por el valor de *save_steps* que indiquemos), se irán sobreescribiendo. Si queremos mantenerlos para el posterior análisis deberemos lanzar el script [copy_checkpoints.py](src/main/python/copy_checkpoints.py), que creará una carpeta en _res_ donde moverá los checkpoints nada más sean creados. Para lanzarlo, se usará también _nohup_:
```
nohup python -u  /datassd/proyectos/tfm-alvaro/tfm-modelos-multimodales/src/main/python/copy_checkpoint.py > copy_check.txt &
```

Para detener cualquiera de los procesos activos en _nohup_, tomar el número de identificación del proceso y escribir en terminal
```
kill -9 [ID]
```

Es importante tener en cuenta que para el posterior correcto funcionamiento de los scripts habrá que renombrar las carpetas *train_x_x_x* creadas por el script por otro nombre que contenga las palabras *llava* y *lora* explícitamente (como *llava_lora_train_x_x_x*), pues de no ser así no podrán ser mergeados los checkpoints. Este proceso de mergeo podremos llevarlo a cabo con el script [merge_lora_weights.py](/datassd/proyectos/tfm-alvaro/tfm-modelos-multimodales/model/LLaVA/scripts/merge_lora_weights.py) que proporciona el propio modelo de LLaVA.

Además, será necesario que el checkpoint que queramos mergear contenga un archivo **config.json**. De no ser así, AAAAAAAAAAA

Para ello, elegiremos una nueva carpeta _prueba_ donde guardar los resultados y escribiremos por terminal

```
python model/LLaVA/scripts/merge_lora_weights.py --model-path res/llava_lora_train_x_x_x/checkpoint-x --model-base liuhaotian/llava-v1.5-7b --save-model-path prueba/
```

## Inferencia

Una vez el modelo ha sido entrenado, y antes de generar predicciones, debemos hacerle unas cuantas cosas antes.
1. En primer lugar deberemos entrenarlo desde los checkpoints una pequeña fracción de época más, para generar así el archivo _config.json_ que guarda la configuración del modelo. Para ello se usará el script ---.
2. Con el archivo _config.json_ podemos proceder a mergear los pesos de **LoRA** con el script ---.

Ahora el modelo está listo para la inferencia. Para probarlo puede lanzarse en **Gradio** mediante los siguientes comandos:

```
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

```
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

```
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path [...] --model-base liuhaotian/llava-v1.5-7b
```

Para probarlo desde la terminal puede lanzarse el _llava.serve.cli_:

```
python -m llava.serve.cli --model-path [...] --model-base liuhaotian/llava-v1.5-7b --image-file [...]
```

Para generar un archivo JSON con todas las predicciones del modelo sobre el conjunto de prueba, deberá accederse al script [generate.py]{} dentro de la carpeta _evaluation_ e introducir la ruta a nuestro modelo en el ```"--model-path"``` y ejecutar. El script generará un archivo _final.json_ con la misma estructura que los datasets. Se deberá repetir

## Evaluación

Una vez ha terminado el fine-tuning y el modelo ha sido entrenado con el conjunto de entrenamiento, es momento de pasar a la evaluación de resultados, usando el conjunto de prueba. El modelo finetuneado ocupa menos memoria en GPU, por lo que ya no supondrá tanto problema.

Para la evaluación de los resultados usaremos una métrica de error específica para las tareas multimodales de **captioning** y **VQA**:

    - [**CIDEr**](https://arxiv.org/pdf/1411.5726) (Consensus-based Image Description Evaluation):
$$CIDEr_n(c_i,S_i)=\dfrac{1}{m}\sum_i\dfrac{g^n(c_i)\cdot g^n(s_{i,j})}{||g^n(c_i)||\cdot||g^n(s_{i,j})},$$
$$g_k(s_{ij})=\dfrac{h_k(s_{ij})}{\sum_{w_j\in\Omega}h_l(s_{ij})}\log\left(\dfrac{|I|}{\sum_{I_p\in I}\min(1,\sum_qh_k(s_{pq}))}\right).$$


### Scripts de evaluación
La carpeta _evaluation_ contiene una serie de scripts que deberemos ir ejecutando para poder llevar a cabo la evaluación de nuestro modelo en función de la métrica **CIDEr**.

```
root
├── data
├── model
├── res
├── src
│   └── main
│       └── python
│           └── evaluation
│               ├── check_df.py
│               ├── cider_scorer.py
│               ├── cider.py
│               ├── custom_df.py
│               └── evaluation.py
└── wandb
```

1. En primer lugar, ejecutaremos el script [custom_df.py](src/main/python/evaluation/custom_df.py). Este programa nos creará un archivo **df.p** que contiene un diccionario con las frencuencias de documentos calculadas a partir de los conjuntos de referencia.
2. Como el archivo **df.p** es binario, si queremos visualizarlo podemos ejecutar el script [check_df.py](src/main/python/evaluation/check_df.py), que mostrará 50 entradas del archivo, donde cada entrada es una tupla de un n-grama y su frecuencia de documento. Además, proporcionará la longitud de referencia logarítmica del conjunto de datos de referencia.
3. Una vez tenemos el diccionario de frecuencias, ejecutar el script [evaluation.py](src/main/python/evaluation/evaluation.py) para obtener la métrica **CIDEr**. Habrá que especificar la ruta de nuestro modelo entrenado.