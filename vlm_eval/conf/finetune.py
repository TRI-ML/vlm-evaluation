from dataclasses import dataclass
from typing import List, Any

@dataclass
class FinetuneReferenceConfig:
    uuid: str
    model: dict
    dataset: dict
    pretrain: dict
    stage: str
    run_id: str
