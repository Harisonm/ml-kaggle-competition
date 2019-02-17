import os
import yaml
filestore_root_dir = "./mlruns" # insert your filestore dir (string) here
experiment_id = 0 # insert your experiment ID (int) here
experiment_dir = os.path.join(filestore_root_dir, str(experiment_id))
for run_dir in [elem for elem in os.listdir(experiment_dir) if elem != "meta.yaml"]:
  meta_file_path = os.path.join(experiment_dir, run_dir,  'meta.yaml')
  with open(meta_file_path) as meta_file:
    if yaml.safe_load(meta_file.read()) is None:
      print("Run data in file %s was malformed" % meta_file_path)
