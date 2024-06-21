import json
from cider import Cider
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

# Cargar datos de prueba y predicciones
with open("/datassd/proyectos/tfm-alvaro/results/data_test.json", 'r') as f:
    data_test = json.load(f)

with open("/datassd/proyectos/tfm-alvaro/data/sets/final.json", 'r') as f:
    predictions = json.load(f)

# ground truths y resultados
gts = {}
res = []

# Formatear los datos de prueba y las predicciones (si no da errores de Assertion o no sé qué)
for item in data_test:
    image_id = item['id']
    gts[image_id] = [conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt']

for item in predictions:
    image_id = item['id']
    caption = [conv['value'] for conv in item['conversations'] if conv['from'] == 'gpt'][0]
    res.append({'image_id': image_id, 'caption': [caption]})

# Aplicar la métrica CIDEr
cider = Cider()
score, scores = cider.compute_score(gts, res)
print(f"CIDEr scores per image: {scores}")
print(f"CIDEr Score: {score}")

# Filter out zero scores and compute the mean of non-zero scores
non_zero_scores = [s for s in scores if s != 0]
mean_non_zero_score = np.mean(non_zero_scores)

print(f"CIDEr non-zero: {mean_non_zero_score}")