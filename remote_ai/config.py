from pathlib import Path
import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("sources", {}).setdefault("cameras", [])
    return cfg
