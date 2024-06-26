import os
import re
import datetime

date_init = datetime.datetime.now()
run_name_dir = 'llava_lora_train_' + date_init.strftime('%Y_%m_%d_%H_%M_%S')
print('Moving results to ' + run_name_dir)

hours_limit = 11
date_end = date_init + datetime.timedelta(hours=hours_limit)

danger_dir = '/datassd/proyectos/tfm-alvaro/res/'
safe_dir = '/datassd/proyectos/tfm-alvaro/res/'

dir_pattern = re.compile(r'^checkpoint-\d+$')

while True:
    # In case there are more than one checkpoint folder for some reason
    for directory_name in os.listdir(danger_dir):
        # Check for directory
        if dir_pattern.match(directory_name):
            danger_path = os.path.join(danger_dir, directory_name)
            safe_path = os.path.join(safe_dir, run_name_dir, directory_name)

            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)

            # Check if the destination directory already exists
            if os.path.exists(safe_path):
                continue

            os.rename(danger_path, safe_path)
            print(f"Moved directory '{directory_name}' from '{danger_path}' to '{safe_path}'")

    if datetime.datetime.now() > date_end:
        print('Closing endless loop')
        break
