from evaluate import load

# BLEU (translation)
bleu = load("bleu")
predictions = ["I really am around thirty six years old"]
references = [
    ["I am thirty six years old", "I am thirty six"]
]
result = bleu.compute(predictions=predictions, references=references)

print("\n\n",result)

# precisions: scores de precisión de cada n-grama (hasta 4)
    # coincidencias / n-gramas en candidato
# bleu: media geométrica de los scores de precisión
    # raiz n-esima del producto de precisiones


# ROUGE
rouge = load("rouge")
predictions = ["I really loved reading the Hunger Games"]
references = ["I loved reading the Hunger Games"]
result = rouge.compute(predictions=predictions, references=references)

print("\n\n",result)

# rouge1: F1-score con unigramas
    # 2[(precision*recall)/(precision+recall)]
# rouge2: F2-score con 2-gramas
# rougeL: FL-score con LCSprecision y LCSrecall
    # LCSprecision = secuencia coincidente más larga en orden relativo no contiguo / palabras en prediccion
    # LCSrecall = secuencia coincidente más larga en orden relativo no contiguo / palabras en referencia


# METEOR
meteor = load("meteor")

predictions = ["I really loved reading the Hunger Games"]
references = ["I loved reading the Hunger Games"]
result = meteor.compute(predictions=predictions, references=references)

print("\n\n",result)

# meteor: M = F_mean*(1-p)
    # F_mean = 10PR/(R+9P) con unigramas
    # penalización: número de chunks en el candidato / unigramas en el candidato
        # si hay varios posibles chunkizaciones, tomar la de mínimo número de chunks
    # p = 0.5*penalización^3

