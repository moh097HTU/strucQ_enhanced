# security/train_frontier.py
from typing import List
from .frontier_filter import FrontierFilter  # relative import

def train_and_save(
    benign_prompts: List[str],
    attack_prompts: List[str],
    out_path: str = "security/frontier.pkl",
    model_name: str = "BAAI/bge-small-en-v1.5",
) -> FrontierFilter:
    ff = FrontierFilter.fit(benign_prompts, attack_prompts, model_name=model_name)
    ff.save(out_path)
    return ff
