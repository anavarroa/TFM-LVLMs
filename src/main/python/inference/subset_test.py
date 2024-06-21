import os
import json
SUBSET_SIZE = 150

# Function to create subsets of the original JSON file
def create_json_subsets(input_json, output_dir, subset_size):
    with open(input_json, 'r') as infile:
        data = json.load(infile)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_subsets = (len(data) + subset_size - 1) // subset_size  # Calculate total number of subsets for zero-padding
    
    for i in range(0, len(data), subset_size):
        subset = data[i:i+subset_size]
        subset_num = i // subset_size + 1
        subset_filename = os.path.join(output_dir, f'filter_{subset_num:02}.json')  # Zero-pad the subset number
        with open(subset_filename, 'w') as outfile:
            json.dump(subset, outfile, indent=4)
        print(f'Created {subset_filename}')

# Function to convert JSON to intermediate txt format
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

# Function to process intermediate txt to final format
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
            # Add an extra "exit" at the end if not already there
            if lines[-1].strip() != "":
                outfile.write("exit\n")
            # Append "stop" at the end
            outfile.write("stop\n")
    print(f"Processed {intermediate_txt} into {final_txt}")

# Main function to execute the tasks
def main():
    input_json = '/datassd/proyectos/tfm-alvaro/data/sets/data_test.json'
    base_path = '/datassd/proyectos/tfm-alvaro/data/imagenes'
    output_dir = '/datassd/proyectos/tfm-alvaro/data/sets/filter'
    
    # Step 1: Create subsets of the JSON file
    create_json_subsets(input_json, output_dir, SUBSET_SIZE)

    # Step 2 and Step 3: Convert each subset JSON into txt and then process to final format
    for subset_file in os.listdir(output_dir):
        if subset_file.endswith('.json'):
            intermediate_txt = os.path.join(output_dir, subset_file.replace('.json', '.txt'))
            final_txt = os.path.join(output_dir, subset_file.replace('.json', '_final.txt'))
            
            # Convert JSON to intermediate txt format
            convert_json_to_txt(os.path.join(output_dir, subset_file), intermediate_txt, base_path)
            
            # Process the intermediate txt to generate final format
            process_intermediate_txt(intermediate_txt, final_txt)
            
            # Delete the intermediate txt file
            os.remove(intermediate_txt)
            print(f'Deleted {intermediate_txt}')

if __name__ == '__main__':
    main()
