import os
import glob
import yaml

def find_files(root_dir, extensions):

    file_paths = []
    for ext in extensions:
        pattern = os.path.join(root_dir, "**", f"*.{ext}") 
        file_paths.extend(glob.glob(pattern, recursive=True))
    return file_paths

def create_yaml_config(root_dir = "knowledge", output_file="config/base_config.yaml"):


    csv_files = find_files(root_dir, ["csv"])
    json_files = find_files(root_dir, ["json"])

    config_data = {
        "knowledge_base": {
            "csv_paths": csv_files,
            "json_paths": json_files,
        }
    }

    try:
        with open(output_file, "w") as yaml_file:
            yaml.dump(config_data, yaml_file, indent=2)
        print(f"YAML configuration saved to {output_file}")
    except Exception as e:
        print(f"Error writing YAML file: {e}")
