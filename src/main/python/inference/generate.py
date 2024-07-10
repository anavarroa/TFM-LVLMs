import subprocess
import json
import os

def interact_with_subprocess(command, input_lines):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        output_lines = []
        
        for input_data in input_lines:
            process.stdin.write(input_data + '\n')
            process.stdin.flush()
            print(f"Sent to subprocess: {input_data}")  # Debugging
        
        # Leer el output del subprocess
        while True:
            line = process.stdout.readline()
            if not line:
                break
            output_lines.append(line.strip())
            print(line.strip())  # Imprimir las líneas para debugging

        # Esperar a que el subprocess termine y devolver código
        process.communicate()
        return output_lines, process.returncode

    finally:
        # Asegurarse de que el proceso ha terminado
        if process.poll() is None:
            process.kill()

def update_json_with_predictions(input_json, predictions_file, output_json):
    # Lectura predicciones de final.txt
    with open(predictions_file, 'r') as f:
        text = f.read()

    # Escisión del texto por "Human: Assistant: "
    predictions = text.split('Human: Assistant: ')[1:]

    # Eliminación de cosas innecesarias
    predictions = [pred.split('\nHuman: Switching to a new image...')[0].strip() for pred in predictions]

    # Carga del conjunto original
    with open(input_json, 'r') as f:
        test_data = json.load(f)

    # Actualización con imágenes
    pred_index = 0
    for entry in test_data:
        for conv in entry['conversations']:
            if conv['from'] == 'gpt':
                if pred_index < len(predictions):
                    conv['value'] = predictions[pred_index]
                    pred_index += 1
                else:
                    print(f"Warning: Not enough predictions for {input_json}. Skipping further updates.")
                    break

    # Guardado del nuevo dataset a final.json
    with open(output_json, 'w') as f:
        json.dump(test_data, f, indent=4)

    print(f"{output_json} has been successfully created with updated predictions.")

def merge_json_files(directory):
    # Lista de todos los archivos JSON y clasificación
    json_files = sorted([filename for filename in os.listdir(directory) if filename.endswith('.json')])

    # Inicialización de una lista vacía
    all_entries = []

    # Iteración sobre cada archivo clasificado
    for filename in json_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            all_entries.extend(data)

    # Escrito en results/final.json
    output_file = os.path.join(directory, 'final.json')
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(all_entries, out, ensure_ascii=False, indent=4)

    print(f'Merged data written to {output_file}')

def main():
    # Data paths
    filter_folder = 'RUTA_A_LA_CARPETA_DE_SUBDATASETS' # Modificar
    results_folder = 'RUTA_A_LA_CARPETA_OUTPUT' # Modificar

    # Creación del archivo si no existe
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Procesamiento de los filter_i_final.txt y json
    input_txt_files = [file for file in os.listdir(filter_folder) if file.endswith('_final.txt')]
    input_txt_files.sort()
    input_json_files = [file.replace('_final.txt', '.json') for file in input_txt_files]

    # Alineación correcta
    assert len(input_txt_files) == len(input_json_files), "Number of TXT and JSON files must match."

    for txt_file, json_file in zip(input_txt_files, input_json_files):
        input_txt = os.path.join(filter_folder, txt_file)
        input_json = os.path.join(filter_folder, json_file)

        # Output paths
        final_txt = os.path.join(results_folder, f"{os.path.splitext(txt_file)[0]}.txt")
        final_json = os.path.join(results_folder, f"{os.path.splitext(json_file)[0]}.json")

        # Comando para inferencia con cli
        command = [
            "python", "RUTA_AL_CLI",
            "--model-path", "RUTA_AL_MODELO_ENTRENADO",
            "--model-base", "liuhaotian/llava-v1.5-7b"
        ] # Modificar

        # Lectura de las líneas
        with open(input_txt, 'r') as file:
            lines = file.readlines()

        input_lines = [line.strip() for line in lines if line.strip()]
        
        print(f"Processing {input_txt} and {input_json}...")

        # Interacción con el subprocess
        output, returncode = interact_with_subprocess(command, input_lines)

        if returncode == 0:
            print("Model loaded and predictions completed successfully.")
        else:
            print("An error occurred while running the model.")

        # Guardado en final.txt
        with open(final_txt, 'w') as final_file:
            final_file.write("\n".join(output))

        print(f"Predictions saved to {final_txt}")

        # Actualización de predicciones y guardado
        update_json_with_predictions(input_json, final_txt, final_json)
    
    # Mergeo de todas las predicciones en un JSON
    merge_json_files(results_folder)

if __name__ == "__main__":
    main()
