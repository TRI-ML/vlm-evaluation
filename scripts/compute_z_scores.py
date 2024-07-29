"""
analyze.py

Analyzes the aggregated metrics to compute the mean and the standard deviation, and adds normalized-z score to the aggregated metrics file for each experiment. 
"""
import os
import re

import draccus
import glob
import json
import numpy as np
import pandas as pd

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tabulate import tabulate


TASK_LIST=["vqa-v2_vqa-v2-full", "vqa-v2_vqa-v2-slim", "gqa_gqa-full", "vizwiz_vizwiz-full", "text-vqa_text-vqa-full", "refcoco_refcoco-full", "ocid-ref_ocid-ref-full"]

METRICS_TO_AVERAGE_OVER = ["accuracy__TextVQA-OCR", "accuracy__TextVQA-Pure", "accuracy", "accuracy__VizWiz-Overall", "accuracy__RefCOCO", "accuracy__RefCOCOg",
                            "accuracy__OCIDRef-All"]

def is_metric(key):
  return re.match(r'^accuracy__', key) or \
            re.match(r'^accuracy$', key)

def get_runid(filepath):
  return filepath.split('/')[-1].split('.')[0]

@dataclass
class AnalyzeEvalsConfig:
  tasks: str = "vqa-v2_vqa-v2-full"    #  comma-seperated (or use 'all')

  results_dir: str = Path("/datasets/mbm/prismtic-vlms/vlm-evaluation/results/aggregated/")

  pattern: str = "*.json"

@draccus.wrap()
def main(cfg: AnalyzeEvalsConfig):

  current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  if cfg.tasks == 'all':
    tasks = TASK_LIST
  else:
    tasks = cfg.tasks.split(',')

  task2scores = {} # dict mapping task to list of scores for computing mean and stds.

  for task in tasks:
    assert task in TASK_LIST, f"Task {task} not found in TASK_LIST {TASK_LIST}"
    task2scores[task] = {}

  filepath2results = {} # dict mapping filepaths to results to update results later with normalized z score.

g  for filepath in glob.glob(os.path.join(cfg.results_dir, cfg.pattern)):
    with open(filepath, 'r') as file:
      results_json = json.load(file)
      filepath2results[filepath] = results_json

      # Add task score from results_json to 
      # task2scores dict. 
      for task in tasks:
        if task not in results_json:
            continue
        
        task_results = results_json[task]
        for key, result in task_results.items():
          if not is_metric(key):
              continue

          if key not in task2scores[task]:
              task2scores[task][key] = {'scores': [], 
                                        'mean': 0., 
                                        'std_dev': 1.
                                        }
      
          assert isinstance(result, float)
          task2scores[task][key]['scores'].append(result)

  for task in tasks:
    for key, score_dict in task2scores[task].items():
        task2scores[task][key]['mean'] = float(np.mean(score_dict['scores']))
        if len(score_dict['scores']) > 1:
          task2scores[task][key]['std_dev'] = float(np.std(score_dict['scores'], dtype=float))

  # Compile filepath to z scores for analysis.
  runid2zscores = {}

  # For each task score, compute z score and add mean and std. dev to
  # the file with the timestamp.
  for filepath, results_json in filepath2results.items():
    # Add timestamp to results_json to track when was last mean and std computed and 
    # on which files. This info will be stored seperately in compiled_results.json.
    results_json['mean_and_std_computed_on'] = current_datetime_str

    run_id = get_runid(filepath)

    aggregated_z_score = 0.
    aggregated_z_scores = []
    for task in tasks:
        if task not in results_json:
            continue
          
        task_results = results_json[task]
        keys = list(task_results.keys())
        for key in keys:
            if not is_metric(key):
                continue
            
            if key not in METRICS_TO_AVERAGE_OVER:
              continue

            result = results_json[task][key]
            mean = task2scores[task][key]['mean']
            std_dev = task2scores[task][key]['std_dev']

            if isinstance(result, float):
                results_json[task][f"{key}-z_score"] = \
                    (result - mean)/std_dev
                results_json[task][f"{key}-mean"] = mean
                results_json[task][f"{key}-std_dev"] = std_dev

                aggregated_z_scores.append(results_json[task][f"{key}-z_score"])

                if run_id not in runid2zscores:
                      runid2zscores[run_id] = {"filepath": filepath}

                runid2zscores[run_id][f'{task}-{key}-zscore'] =  results_json[task][f"{key}-z_score"]
    
    if len(aggregated_z_scores) > 0:
      aggregated_z_score = sum(aggregated_z_scores)/len(aggregated_z_scores)

      runid2zscores[run_id]['aggregated-z_score'] =  aggregated_z_score

      results_json['aggregated-z_score'] = aggregated_z_score

    # Save the updated json with z scores.
    with open(filepath, 'w') as file:
      json.dump(results_json, file, sort_keys=True, indent=2)

  # Pretty print sorted z score table. 
  df = pd.DataFrame(runid2zscores).T

  # sort df values by aggregated score, and sort the column names to bring aggregated score to front.
  df = df.sort_values(by=['aggregated-z_score'], ascending=False).sort_index(axis=1)

  print(tabulate(df[['aggregated-z_score']], headers='keys', tablefmt='fancy_grid'))
  
if __name__=="__main__":
    main()