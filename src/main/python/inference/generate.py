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
            print(f"Sent to subprocess: {input_data}")  # Debugging line
        
        # Read output from subprocess
        while True:
            line = process.stdout.readline()
            if not line:
                break
            output_lines.append(line.strip())
            print(line.strip())  # Print each line for debugging

        # Wait for subprocess to finish and get return code
        process.communicate()
        return output_lines, process.returncode

    finally:
        # Ensure the subprocess is terminated
        if process.poll() is None:
            process.kill()

def update_json_with_predictions(input_json, predictions_file, output_json):
    # Read predictions from final.txt
    with open(predictions_file, 'r') as f:
        text = f.read()

    # Split text by "Human: Assistant: "
    predictions = text.split('Human: Assistant: ')[1:]

    # Remove unnecessary trailing parts (e.g., prompts for next image)
    predictions = [pred.split('\nHuman: Switching to a new image...')[0].strip() for pred in predictions]

    # Load original test_data
    with open(input_json, 'r') as f:
        test_data = json.load(f)

    # Update test_data with predictions
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

    # Save updated test_data to final.json
    with open(output_json, 'w') as f:
        json.dump(test_data, f, indent=4)

    print(f"{output_json} has been successfully created with updated predictions.")

def merge_json_files(directory):
    # Get a list of all JSON files in the directory and sort them
    json_files = sorted([filename for filename in os.listdir(directory) if filename.endswith('.json')])

    # Initialize an empty list to store all entries
    all_entries = []

    # Iterate over each sorted file in the directory
    for filename in json_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            all_entries.extend(data)

    # Write all_entries to results/final.json
    output_file = os.path.join(directory, 'final.json')
    with open(output_file, 'w', encoding='utf-8') as out:
        json.dump(all_entries, out, ensure_ascii=False, indent=4)

    print(f'Merged data written to {output_file}')

def main():
    # Data paths
    filter_folder = '/datassd/proyectos/tfm-alvaro/data/sets/filter/'
    results_folder = '/datassd/proyectos/tfm-alvaro/results/'

    # Ensure results folder exists, create if it doesn't
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Collect all filter_i_final.txt files
    input_txt_files = [file for file in os.listdir(filter_folder) if file.endswith('_final.txt')]
    input_txt_files.sort()  # Ensure files are processed in order if necessary

    # Collect corresponding filter_i.json files
    input_json_files = [file.replace('_final.txt', '.json') for file in input_txt_files]

    # Ensure pairs are aligned correctly
    assert len(input_txt_files) == len(input_json_files), "Number of TXT and JSON files must match."

    for txt_file, json_file in zip(input_txt_files, input_json_files):
        input_txt = os.path.join(filter_folder, txt_file)
        input_json = os.path.join(filter_folder, json_file)

        # Output paths
        final_txt = os.path.join(results_folder, f"{os.path.splitext(txt_file)[0]}.txt")
        final_json = os.path.join(results_folder, f"{os.path.splitext(json_file)[0]}.json")

        # Now run generate.py using input_txt and input_json
        command = [
            "python", "/datassd/proyectos/tfm-alvaro/model/LLaVA/llava/serve/cli.py",
            "--model-path", "/datassd/proyectos/tfm-alvaro/2_prueba_llava_lora_train_128_100_1e-4/",
            "--model-base", "liuhaotian/llava-v1.5-7b"
        ]

        # Read input lines from input_txt
        with open(input_txt, 'r') as file:
            lines = file.readlines()

        input_lines = [line.strip() for line in lines if line.strip()]
        
        print(f"Processing {input_txt} and {input_json}...")

        # Interact with the subprocess
        output, returncode = interact_with_subprocess(command, input_lines)

        if returncode == 0:
            print("Model loaded and predictions completed successfully.")
        else:
            print("An error occurred while running the model.")

        # Save predictions to final.txt
        with open(final_txt, 'w') as final_file:
            final_file.write("\n".join(output))

        print(f"Predictions saved to {final_txt}")

        # Update input_json with predictions and save to final.json
        update_json_with_predictions(input_json, final_txt, final_json)
    
    # Merge all JSON files into results/final.json
    merge_json_files(results_folder)

if __name__ == "__main__":
    main()
