import os
import json
SUBSET_SIZE = 100 # Modificar

# Función para crear subsets de JSON original
def create_json_subsets(input_json, output_dir, subset_size):
    with open(input_json, 'r') as infile:
        data = json.load(infile)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_subsets = (len(data) + subset_size - 1) // subset_size  # Calculate total number of subsets for zero-padding
    
    for i in range(0, len(data), subset_size):
        subset = data[i:i+subset_size]
        subset_num = i // subset_size + 1
        subset_filename = os.path.join(output_dir, f'filter_{subset_num:03}.json')  # Formateo de tres dígitos
        with open(subset_filename, 'w') as outfile:
            json.dump(subset, outfile, indent=4)
        print(f'Created {subset_filename}')

# Conversión de JSON a formato intermedio
def convert_json_to_txt(input_json, output_txt, base_path):
    with open(input_json, 'r') as infile:
        data = json.load(infile)
    
    with open(output_txt, 'w') as outfile:
        for item in data:
            image_path = os.path.join(base_path, item['image'])
            outfile.write(f"image: {image_path}\n")
            for conversation in item['conversations']:
                if conversation['from'] == 'human':
                    prompt = conversation['value'].replace('<image>\n', '')
                    outfile.write(f"prompt: {prompt}\n")
            outfile.write("\n")

# Procesamiento al formato final
def process_intermediate_txt(intermediate_txt, final_txt):
    with open(intermediate_txt, 'r') as infile:
        with open(final_txt, 'w') as outfile:
            lines = infile.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith("image:"):
                    outfile.write(line.split("image:")[1].strip() + "\n")
                elif line.startswith("prompt:"):
                    outfile.write(line.split("prompt:")[1].strip() + "\n")
                elif line == "":
                    outfile.write("exit\n")
            # Añadido del exit
            if lines[-1].strip() != "":
                outfile.write("exit\n")
            # Añadido de stop
            outfile.write("stop\n")
    print(f"Processed {intermediate_txt} into {final_txt}")

# Ejecución de las tareas
def main():
    input_json = 'RUTA_AL_CONJUNTO_DE_PRUEBA' # Modificar
    base_path = 'RUTA_A_LA_CARPETA_DE_IMAGENES' # Modificar
    output_dir = 'RUTA_A_LA_CARPETA_OUTPUT' # Modificar
    
    # Creación de subconjuntos
    create_json_subsets(input_json, output_dir, SUBSET_SIZE)

    # Conversión al formato final
    for subset_file in os.listdir(output_dir):
        if subset_file.endswith('.json'):
            intermediate_txt = os.path.join(output_dir, subset_file.replace('.json', '.txt'))
            final_txt = os.path.join(output_dir, subset_file.replace('.json', '_final.txt'))
            
            convert_json_to_txt(os.path.join(output_dir, subset_file), intermediate_txt, base_path)

            process_intermediate_txt(intermediate_txt, final_txt)
            
            # Eliminación de archivos intermedios
            os.remove(intermediate_txt)
            print(f'Deleted {intermediate_txt}')

if __name__ == '__main__':
    main()
