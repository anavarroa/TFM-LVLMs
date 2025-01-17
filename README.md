# LLaVA 1.5-7B para tareas de Image Captioning y VQA con un dataset de Remote Sensing personalizado.

<div align="center">
  <img src="imagen.jpeg" alt="LLaVA 1.5 para RS" width="25%">
</div>

1. [Conjunto de datos](#conjunto-de-datos)
   - [Descarga del dataset](#descarga-del-dataset)
   - [Partición](#partición)
2. [Descarga del modelo](#descarga-del-modelo)
3. [Carga del modelo preentrenado](#carga-del-modelo-preentrenado)
   - [Rutas](#rutas)
   - [Parámetros](#parámetros)
   - [Parámetros Importantes](#parámetros-importantes)
4. [Entrenamiento](#entrenamiento)
   - [Errores](#errores)
   - [Éxito](#éxito)
5. [Inferencia](#inferencia)
   - [Gradio](#gradio)
   - [Cli](#cli)
   - [Automatización](#automatización)
   - [Caso particular](#caso-particular)
6. [Evaluación](#evaluación)
   - [Scripts de evaluación](#scripts-de-evaluación)

## Conjunto de datos

El dataset de entrenamiento consta de datos de varios datasets de _Remote Sensing_ (RS): **NWPU**, **RSICD**, **LR** y **DOTA**. El script [data_preparation](src/main/python/data_preparation.py) del que se habla abajo se encargará de parte de la descarga y preparación del dataset.

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
- **Conjunto de prueba** (*data_test*): datos que se usarán para comprobar la calidad del modelo ya entrenado, durante la evaluación de resultados.

El script [data_partition](src/main/python/data_partition.py) llevará a cabo la partición mencionada del conjunto de datos, creando tres archivos JSON correspondientes a los dos conjuntos a considerar, y los ubicará en una carpeta _sets_ dentro de _data_:

```
data
├── imagenes
├── sets
│   ├── data_train.json
│   └── data_test.json
└── dataset.json
```

El porcentaje de datos en cada conjunto es por defecto 70% _TRAIN_ y 30% _TEST_ (con semilla fijada). Si se desea una partición distinta o se va a considerar un conjunto de validación, pueden modificarse estos porcentajes desde el script, en la función *crear_datasets(directorio, train_ratio, test_ratio)*:
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
- **TEST_DATA_PATH**: ruta al conjunto de prueba *data/sets/data_test.json*.
- **IMAGE_FOLDER**: ruta a la carpeta _data/imagenes_.
- **OUTPUT_DIR**: ruta a la carpeta donde queremos guardar los resultados.

### Parámetros

Además de las rutas, es crucial configurar los parámetros del modelo. El siguiente stream ejecutará el entrenamiento con __LoRA__, una técnica de fine-tuning más eficiente basada en la utilización de matrices de menor rango. Los parámetros ajustables son los que aparecen en él.

```
finetune_script = f'''
{DEEPSPEED_SCRIPT} \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed {DEEPSPEED_JSON} \
    --model_name_or_path {MODEL_NAME} \
    --version v1 \
    --data_path {TRAIN_DATA_PATH} \
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
    --num_train_epochs 10.1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
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

Se recomienda mantener el tamaño global de lota (*global_batch_size*) constante durante todos los entrenamientos, calculado como
$$global\_batch\_size=per\_device\_train\_batch\_size\times gradient\_accumulation\_steps\times num\_gpus.$$

En caso de lanzarlo sin **LoRA**, deberá eliminarse toda la primera línea del stream:
```
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
```
Esto, no obstante, puede llegar a dar problemas por falta de memoria en GPU.

El stream anterior utiliza **DeepSpeed**, una librería de optimización de Deep Learning que proporciona una amplia gama de funciones para acelerar el entrenamiento de modelos grandes en GPU y sistemas distribuidos.

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
nohup deepspeed model/LLaVA/llava/train/train_mem.py     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5     --deepspeed model/LLaVA/scripts/zero3.json     --model_name_or_path liuhaotian/llava-v1.5-7b     --version v1     --data_path .../data/dataset.json     --image_folder .../data/imagenes     --vision_tower openai/clip-vit-large-patch14-336     --mm_projector_type mlp2x_gelu     --mm_vision_select_layer -2     --mm_use_im_start_end False     --mm_use_im_patch_token False     --image_aspect_ratio pad     --group_by_modality_length True     --bf16 True     --output_dir .../res     --num_train_epochs 0.05     --per_device_train_batch_size 16     --per_device_eval_batch_size 4     --gradient_accumulation_steps 1     --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 50000     --save_total_limit 1     --learning_rate 2e-4     --weight_decay 0.     --warmup_ratio 0.03     --lr_scheduler_type "cosine"     --logging_steps 1     --tf32 True     --model_max_length 2048     --gradient_checkpointing True     --dataloader_num_workers 4     --lazy_preprocess True     --report_to wandb > log.txt &
```

Obtener un error pasado este punto es muy posiblemente debido a una falta de memoria en GPU (reducir *batch_size* en tal caso).

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
nohup python -u  .../src/main/python/copy_checkpoint.py > copy_check.txt &
```

Es importante tener en cuenta que para el posterior correcto funcionamiento de los scripts habrá que renombrar las carpetas *train_x_x_x* creadas por el script por otro nombre que contenga las palabras *llava* y *lora* explícitamente (como *llava_lora_train_x_x_x*), pues de no ser así no podrán ser mergeados los checkpoints. Este proceso de mergeo podremos llevarlo a cabo con el script [merge_lora_weights.py](./model/LLaVA/scripts/merge_lora_weights.py) que proporciona el propio modelo de LLaVA.

También hay que tener en cuenta que la aproximación llevada a cabo en el entrenamiento implica que el resultado no tiene la estructura de un modelo de Hugging Face, sino de checkpoint, por lo que es necesario proseguir con el entrenamiento para obtener un objeto modelo. Para realizar este proceso con todos los checkpoints almacenados de diferentes entrenamientos, se utiliza el script [checkpoint_to_lora.sh](src/main/bash/checkpoint_to_lora.sh), que esencialmente realiza un entrenamiento igual al descrito en [Entrenamiento](#entrenamiento) para todos los checkpoints almacenados pero con únicamente una fracción (configurable) de época. Los resultados (modelos LORA) se almacenan en la carpeta *lora_res*.

Una vez el modelo ha sido entrenado, y antes de generar predicciones, se puede proceder a realizar inferencia.


## Inferencia

Ahora el modelo está listo para la inferencia, ya podemos generar predicciones para el conjunto de datos de prueba. Dependiendo del objetivo se pueden llevar a cabo varios métodos de inferencia, dependiendo de si se realiza con el modelo base (con estructura de modelo de Hugging Face) o con los modelos LoRA entrenados (que tienen que mergearse al modelo base).

### Gradio

Para hacer pequeñas pruebas con el modelo puede lanzarse en **Gradio** mediante los siguientes comandos:

```
python -m llava.serve.controller --host 0.0.0.0 --port 10000
```

```
python -m llava.serve.gradio_web_server --controller http://localhost:10000 --model-list-mode reload
```

```
CUDA_VISIBLE_DEVICES=0 python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path [...] --model-base liuhaotian/llava-v1.5-7b
```

### Cli

Para probarlo desde la terminal puede lanzarse el _llava.serve.cli_:

```
python -m llava.serve.cli --model-path [...] --model-base liuhaotian/llava-v1.5-7b --image-file [...]
```

Esto funciona gracias al script __cli.py__ proporcionado por el propio modelo, sin embargo no permite la automatización de la generación de predicciones. Es por ello que para proseguir es necesario sustituir su código con el del [cli.py](cli.py) modificado que se encuentra en el repositorio. El funcionamiento del script ha sido cambiado para satisfacer las necesidades del proyecto, incorporando los siguientes cambios:
- Capacidad de introducir la imagen como input en lugar de como parámetro.
- Posibilidad de analizar y predecir sobre varias imágenes con una única carga del modelo. Escribiendo "exit" en un prompt, se podrá introducir la ruta de una nueva imagen.
- Modificaciones necesarias para poder ser lanzado como subprocess.
- Interrupción de la ejecución al recibir la orden "stop".

### Automatización

El siguiente paso será preparar los conjuntos sobre los que el modelo realizará la inferencia. Por la naturaleza del subprocess, existe un delay entre prompt y prompt mucho más pronunciado que por terminal, y el tiempo de carga del modelo crece exponencialmente a más imágenes se pretenda analizar. Por ello, se realizará una partición del conjunto de prueba en subconjuntos de unas cuantas imágenes cada uno. Esto se consigue con el script [subset_test.py](src/main/python/inference/subset_test.py), en el que podemos indicar el tamaño de los subconjuntos (100 por defecto). Los subconjuntos se guardarán en una carpeta _filter_ dentro de _sets_. Una vez realizada la partición no debe modificarse para no alterar resultados entre modelos distintos.

Con estas modificaciones, ya podemos usar nuestro conjunto de prueba para generar predicciones. Para obtener un archivo JSON con todas las predicciones del modelo, deberá accederse al script [generate.py](src/main/python/inference/generate.py) dentro de la carpeta _evaluation_ e introducir la ruta a nuestro modelo en el ```"--model-path"``` y ejecutar. Conviene también modificar la ruta del dataset de ser errónea (*filter_folder*) y de la carpeta donde queramos alojas las predicciones (*results_folder*). Se recomienda que esta última ruta sea del tipo "_.../results/nombre-del-modelo/subsets_". El script generará un archivo *filter_final.json* para cada subconjunto y mergeará todos ellos al terminar, creando un archivo *final.json* con la misma estructura que los datasets y las predicciones. Se puede sacar este JSON fuera de la carpeta, a "_.../results/nombre-del-modelo_" para su fácil localización en el futuro.

Para lanzar el script lo mejor es usar de nuevo _nohup_, pues tardará unas cuantas horas (en torno a $15$ en nuestro caso):

```
nohup python src/main/python/evaluation/generate.py > gen_log.txt &
```
El proceso deberá repetirse tantas veces como de modelos entrenados se disponga, modificando en cada caso la orden _--model-path_ en el comando del script. Se recomienda almacenar o renombrar los resultados para cada modelo. Para hacer inferencia con el modelo base de **LLaVA** y poder así comparar, mirar el siguiente subapartado.

**NOTA**: si el proceso tarda demasiado puede deberse a un problema de GPU, por lo que se recomienda no paralelizar y ocupar una única GPU lo más vacía posible. Si es necesario puede disminuirse el tamaño del conjunto de prueba mediante el script [filter_test.py](src/main/python/inference/filter_test.py), si el tiempo de carga o inferencia fuese demasiado grande. También pueden lanzarse varias predicciones simultáneas para economizar el tiempo, en distintas **GPUs** desocupadas o por varios servidores.

```
data
├── imagenes
├── sets
│   ├── filter
│   │   └── ...
│   ├── filtered_test.json
│   ├── data_train.json
│   └── data_test.json
└── dataset.json
results
├── nombre_modelo_1
│   ├── subsets
│   │   └── ...
│   └── final.json
├── nombre_modelo_2
│   ├── subsets
│   │   └── ...
│   └── final.json
└── ...
```

### Caso particular
Para la generación de predicciones con el modelo base deberán hacerse unas ligeras modificaciones al script de inferencia [generate.py](src/main/python/inference/generate.py). Este modelo está construido de forma que ciertas palabras clave cambian, luego se deberán sustituir en el código las palabras "Human" y "Assistant" por "USER" y "ASSISTANT" respectivamente para que se copian de forma correcta las predicciones.

De igual manera, se deberán modificar los parámetros del comando, poniendo en este caso como _--model-path_ el del modelo base y eliminando la orden _--model-base_:

```
command = [
    "python", "/datassd/proyectos/tfm-alvaro/model/LLaVA/llava/serve/cli.py",
    "--model-path", "liuhaotian/llava-v1.5-7b"
]
```

En general, es importante tener en cuenta que pueden aparecer problemas de predicción causados por ciertos parámetros. Por ejemplo un modelo entrenado con learning rate $5e-3$ puede que no logre procesar el dataset. Además, modelos mal entrenados, con pesos de LoRa mal mergeados o problemas de loss nula generados tras el mini-entrenamiento pueden provocar predicciones inválidas, monosilábicas, simbólicas o absurdas. En este último caso, comprobar el archivo *trainer_state.json* del modelo en cuestión para comprobar el estado de sus parámetros en los últimos steps.


## Evaluación

Una vez ha terminado el fine-tuning y se dispone de las predicciones, es momento de pasar a la evaluación de resultados. Para la evaluación de los resultados usaremos una métrica de error específica para las tareas multimodales para visión y lenguaje, **captioning** y **VQA**: [**CIDEr**](https://arxiv.org/pdf/1411.5726) (Consensus-based Image Description Evaluation):
$$CIDEr_n(c_i,S_i)=\dfrac{1}{m}\sum_i\dfrac{g^n(c_i)\cdot g^n(s_{i,j})}{||g^n(c_i)||\cdot||g^n(s_{i,j})},$$
$$g_k(s_{ij})=\dfrac{h_k(s_{ij})}{\sum_{w_j\in\Omega}h_l(s_{ij})}\log\left(\dfrac{|I|}{\sum_{I_p\in I}\min(1,\sum_qh_k(s_{pq}))}\right).$$


### Scripts de evaluación
La carpeta _evaluation_ en _src/main/python_ contiene una serie de scripts que permiten el cálculo correcto de la métrica **CIDEr**:

```
root
├── data
├── model
├── res
├── src
│   └── main
│       └── python
│           └── evaluation
│               ├── cider_scorer.py
│               ├── cider.py
│               └── evaluation.py
└── wandb
```

Para la evaluación, simplemente se indicarán las rutas de los JSONs (conjunto de prueba y predicciones) en el script [evaluation.py](src/main/python/evaluation/evaluation.py), y se ejecutará. Como resultado se mostrará una colección de métricas de error individuales para cada predicción y una puntuación de la métrica **CIDEr** general calculada como la media de todas las anteriores. El proceso creará una carpeta paralela llamada *pycache* que puede ser eliminada una vez obtenida la métrica.

Se ha añadido al script de evaluación una funcionalidad para obtener la media de los valores no nulos de la métrica **CIDEr**. Esto se debe a que dichos valores no suelen considerarse válidos o realistas, y desestimarlos da una idea más fidedigna del desempeño del modelo. Estos valores nulos son causados por conjuntos de datos cuyos prompts arrojan respuestas escuetas, booleanas o numéricas. Este es el caso de los datasets **LR** y **DOTA**, cuyas predicciones suelen ser $0.0$ independientemente de la calidad del modelo.

La evaluación no crea ningún archivo, simplemente mostrará el valor de la métrica por terminal, y no tardará más de diez segundos, luego este último paso no supondrá ningún inconveniente de tiempo a diferencia del entrenamiento y la inferencia.


