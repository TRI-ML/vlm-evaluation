# export model_dir=$model_id+stage-finetune+x7

# # List of tasks
# tasks=("vqa-v2-full" "gqa-full" "vizwiz-full" "text-vqa-full" "refcoco-full" "ocid-ref-full" "pope-full")

# # Loop through each task and run the evaluation and scoring scripts
# for task in "${tasks[@]}"; do
#     echo "Running task: $task"
#     accelerate launch --num_processes=8 ../vlm-evaluation/scripts/evaluate.py --model_id $model_id --model_dir runs/$model_dir/ --dataset.type $task --dataset.root_dir /datasets/prismatic-vlms/vlm-evaluation/
# done


# for task in "${tasks[@]}"; do
#         echo "Evaluating task: $task"
#     python ../vlm-evaluation/scripts/score.py --model_id $model_id --dataset.type $task --dataset.root_dir /datasets/prismatic-vlms/vlm-evaluation/ --results_dir results
# done
import os
import subprocess
import json
import argparse
import draccus
import torch
from dataclasses import dataclass, field, asdict
from vlm_eval.util import set_seed
from pathlib import Path
from typing import Union, Optional
import uuid
from datetime import datetime
import fsspec
import time

from prismatic.util.distributed_utils import world_info_from_env
from prismatic.util.file_utils import llm_backbone_id_to_mbm_configs
from vlm_eval.models import load_vlm
from vlm_eval.conf.datasets import DatasetConfig, DatasetRegistry
from scripts.evaluate import EvaluationConfig, evaluate_after_parse
from scripts.score import ScoreConfig, score_after_parse
from vlm_eval.conf import FinetuneReferenceConfig
from prismatic.conf import PretrainReferenceConfig

TASK_LIST=["vqa-v2-full", "vqa-v2-slim", "gqa-full", "vizwiz-full", "text-vqa-full", "refcoco-full", "ocid-ref-full", "pope-full"]

@dataclass
class EvalRunnerConfig:
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.AI2D_FULL.dataset_id)
    )

    # Eval Configs
    tasks: str = "vqa-v2-full"                      # comma-separated (or use 'all')
     
    # === Model Parameters =>> Prismatic ===
    model_family: str = "prismatic"                 # Model family to load from in < `prismatic` | `llava-v15` | ... >
    model_id: Optional[str] = None
    model_dir: Optional[str] = None                 # Can be local or S3

    # Inference Parameters
    device_batch_size: int = 1                      # Device Batch Size set to 1 until LLaVa/HF LLaMa fixes bugs!
    num_workers: int = 2                            # Number of Dataloader Workers (on each process)

    # Artifact Parameters
    dataset_dir: Path = Path(                       # Path to dataset directory
        "/datasets/prismatic-vlms/vlm-evaluation/"
    )
    results_dir: Path = Path(                       # Path to results directory (writing predicted output, metrics)
        "results"
    )
    remote_sync: str = None
    remote_sync_expdata: str = "s3://tri-ml-datasets/mbm/exp_data"
    remote_sync_frequency: int = 300

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")  # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                  # Random Seed (for reproducibility)

    # Scoring
    score_only: bool = False                        # Assume generation is done and only do scoring
    score_only_s3_dir: str = None

    def __post_init__(self) -> None:
        self.run_dir = self.model_dir


def get_real_name(model_id):
    fs = fsspec.filesystem('s3')
    folders_to_check = [
        "s3://tri-ml-datasets/prismatic/sedrick.keh",
        "s3://tri-ml-datasets/prismatic/jean.mercat",
        "s3://tri-ml-datsaets/openlm/mbm_jean"
    ]
    for i in folders_to_check:
        dirlist = fs.listdir(i)
        dirlist = [f"s3://{j['Key']}" for j in dirlist]
        curr = f"{i}/{model_id}"
        if curr in dirlist:
            curr_full = f"s3://{curr}/config.json"
            with fs.open(curr_full, 'r') as f:
                curr_full_data = json.load(f)
                real_name = curr_full_data['model']['llm_backbone_id']
                if real_name[-1]=="/":
                    real_name = real_name[:-1]
                real_name = f"llava-multimodal+{real_name.rsplit('/', 1)[-1]}+stage-finetune+x7" 
                return real_name
    return model_id


def prismatic_run_name_to_pretrain_config(remote_sync, remote_sync_expdata, model_id):
    mbm_configs = llm_backbone_id_to_mbm_configs(model_id)
    return asdict(PretrainReferenceConfig(
        name=mbm_configs['name'],
        dataset_name=mbm_configs['dataset_name'],
        dataset_uuid=mbm_configs['dataset_uuid'],
        hyperparameters=mbm_configs['hyperparameters'],
        checkpoint_url=mbm_configs['checkpoint_url'],
        results=mbm_configs['results'],
        params_url=mbm_configs['params_url'],
        uuid=mbm_configs['uuid'],
        creation_date=mbm_configs['creation_date'],
        failed=mbm_configs['failed'],
        error=mbm_configs['error'],
        dataset_weights=mbm_configs['dataset_weights'],
        openlm_text_pretrained=mbm_configs.get('openlm_text_pretrained', None),
    )) 
    

def prismatic_run_name_to_finetune_config(remote_sync, remote_sync_expdata, model_id):
    cmd = f"aws s3 cp {remote_sync_expdata}/models/prismatic/{model_id}.json -"
    print("cmd: ", cmd)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if len(stdout) > 0:
        prismatic_finetune_configs = json.loads(stdout)
    else:
        # Check the case where the model was named in a non-default way
        real_name = get_real_name(model_id)
        cmd = f"aws s3 cp {remote_sync_expdata}/models/prismatic/{real_name}.json -"
        print("cmd: ", cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if len(stdout) > 0:
            prismatic_finetune_configs = json.loads(stdout)
        else:
            return False


    return asdict(FinetuneReferenceConfig(
        uuid=prismatic_finetune_configs.get('uuid', None),
        model=prismatic_finetune_configs.get('model', None),
        dataset=prismatic_finetune_configs.get('dataset', None),
        pretrain=prismatic_finetune_configs.get('pretrain', None),
        stage=prismatic_finetune_configs.get('stage', None),
        run_id=prismatic_finetune_configs.get('run_id', None)
    )) 


def try_to_find_existing_eval_json(remote_sync, remote_sync_expdata, results_dir, model_id):
    aggregated_scores = {}
    if remote_sync_expdata is not None:
        json_path_remote = os.path.join(remote_sync_expdata, "eval", f"eval_{model_id}.json")
        cmd = f"aws s3 cp {json_path_remote} -"
        print("cmd 1: ", cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if len(stdout) > 0:
            aggregated_scores = json.loads(stdout)
            if len(aggregated_scores) > 0:
                return aggregated_scores
    if remote_sync is not None:
        aggregated_path_remote = os.path.join(remote_sync, results_dir, "aggregated", f"{model_id}.json")
        cmd = f"aws s3 cp {aggregated_path_remote} -"
        print("cmd 2: ", cmd)
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if len(stdout) > 0:
            aggregated_scores = json.loads(stdout)
            if len(aggregated_scores) > 0:
                return aggregated_scores
    return aggregated_scores


@draccus.wrap()
def main(cfg: EvalRunnerConfig):
    assert cfg.model_id is not None and cfg.model_dir is not None
    if cfg.score_only:
        assert cfg.score_only_s3_dir is not None
    if cfg.remote_sync is not None:
        if cfg.remote_sync[-1] == "/":
            cfg.remote_sync = cfg.remote_sync[:-1]
    if cfg.remote_sync_expdata is not None:
        if cfg.remote_sync_expdata[-1] == "/":
            cfg.remote_sync_expdata = cfg.remote_sync_expdata[:-1]

    # Parse tasks
    datasets = []
    if cfg.tasks == "all":
        tasks = TASK_LIST
    else:
        tasks = cfg.tasks.split(',')
    for t in tasks:
        assert t in TASK_LIST, f"Task {t} not found in TASK_LIST {TASK_LIST}"
        datasets.append(DatasetConfig.get_choice_class(t))

    # Load VLM
    cfg.run_dir = cfg.model_dir
    set_seed(cfg.seed)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    if not cfg.score_only:
        vlm = load_vlm(cfg.model_family, cfg.model_id, cfg.run_dir, hf_token=hf_token, ocr=cfg.dataset.ocr)

    # Get existing scores
    aggregated_scores = try_to_find_existing_eval_json(cfg.remote_sync, cfg.remote_sync_expdata, cfg.results_dir, cfg.model_id)
    print("aggregated scores: ", aggregated_scores)
    aggregated_path = os.path.join(cfg.results_dir, "aggregated", f"{cfg.model_id}.json")
    if cfg.remote_sync is not None:
        aggregated_path_remote = os.path.join(cfg.remote_sync, cfg.results_dir, "aggregated", f"{cfg.model_id}.json")
    _, global_rank, _ = world_info_from_env()
    if global_rank == 0:
        os.makedirs(os.path.join(cfg.results_dir, "aggregated"), exist_ok=True)
        aggregated_scores["name"] = aggregated_scores.get("name", cfg.model_id)
        aggregated_scores["uuid"] = aggregated_scores.get("uuid", str(uuid.uuid4()))
        aggregated_scores["creation_date"] = aggregated_scores.get("creation_date", datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    torch.distributed.barrier()

    if cfg.model_dir.startswith("(open"):
        aggregated_scores["pretrain"] = aggregated_scores.get("pretrain", prismatic_run_name_to_pretrain_config(cfg.remote_sync, cfg.remote_sync_expdata, cfg.model_id))
        assert aggregated_scores["pretrain"] is not None
    else:
        aggregated_scores["finetune"] = aggregated_scores.get("finetune", prismatic_run_name_to_finetune_config(cfg.remote_sync, cfg.remote_sync_expdata, cfg.model_id))
        assert aggregated_scores["finetune"] is not None
    torch.distributed.barrier()

    for dataset in datasets:
        task_name_full = dataset.dataset_id
        task_name_short = task_name_full[:-5]
        aggregated_name = f"{task_name_short}_{task_name_full}"
        if aggregated_name in aggregated_scores:
            if "FAILED" in aggregated_scores[aggregated_name]:
                pass
            else:
                print(f"{aggregated_name} in {aggregated_path}! Skipping.")
                continue

        cfg.dataset = dataset
        cfg.dataset.root_dir = Path(cfg.dataset_dir)
        print(f"Now evaluating: {dataset.dataset_id}")
        if not cfg.score_only:
            evaluate_after_parse(cfg=cfg, vlm=vlm)

        torch.distributed.barrier()
        _, global_rank, _ = world_info_from_env()
        if global_rank == 0:
            sc = ScoreConfig()
            sc.dataset = dataset
            sc.dataset.root_dir = Path(cfg.dataset_dir)
            sc.model_id = cfg.model_id
            sc.score_only_s3_dir = cfg.score_only_s3_dir
            print(f"Now scoring: {dataset.dataset_id}")
            score_after_parse(cfg=sc)

        if global_rank == 0 and cfg.remote_sync is not None:
            print(f"Syncing results to {os.path.join(cfg.remote_sync, cfg.results_dir)}:")
            task_name_full = dataset.dataset_id
            task_name_short = task_name_full[:-5]

            local_results_path = os.path.join(cfg.results_dir, task_name_short, task_name_full, cfg.model_id)
            s3_results_path = os.path.join(cfg.remote_sync, cfg.results_dir, task_name_short, task_name_full, cfg.model_id)
            if local_results_path != s3_results_path:
                cmd = f"aws s3 cp {local_results_path} {s3_results_path} --recursive"
                proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"{task_name_short} remote sync finished.")
            
            # Updated aggregated scores
            retries = 3
            while retries > 0:
                try:
                    cmd = f"aws s3 cp {os.path.join(cfg.remote_sync, cfg.results_dir, task_name_short, task_name_full, cfg.model_id, 'metrics.json')} -"
                    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = proc.communicate()
                    curr_results = json.loads(stdout)
                    break
                except:
                    retries -= 1
                    time.sleep(5)
                    if retries == 0:
                        curr_results = {"summary": f"FAILED, could not load {os.path.join(cfg.remote_sync, cfg.results_dir, task_name_short, task_name_full, cfg.model_id, 'metrics.json')}"}

            aggregated_scores[aggregated_name] = curr_results["summary"]
            with open(aggregated_path, 'w') as f:
                json.dump(aggregated_scores, f, indent=4)
            cmd = f"aws s3 cp {aggregated_path} {aggregated_path_remote}"
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("aggregate remote sync finished.")

        torch.distributed.barrier()

    # Sync to exp_data
    _, global_rank, _ = world_info_from_env()
    if global_rank == 0 and cfg.remote_sync is not None:
        with open(aggregated_path, 'w') as f:
            json.dump(aggregated_scores, f, indent=4)
        cmd = f"aws s3 cp {aggregated_path} {aggregated_path_remote}"
        print("aggregate remote sync cmd:", cmd)
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("aggregate remote sync finished.")
            
        cfg.remote_sync_expdata = cfg.remote_sync_expdata[:-1] if cfg.remote_sync_expdata[-1]=="/" else cfg.remote_sync_expdata
        cmd = f"aws s3 cp {aggregated_path_remote} {cfg.remote_sync_expdata}/eval/eval_{cfg.model_id}.json"
        print("final sync cmd: ", cmd)
        proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("final sync finished.")
    torch.distributed.barrier()


if __name__=="__main__":
    main()