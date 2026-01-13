import json
import os
from utils.utils import clean_text, split_data
from utils.logging_config import logger

RAW_PATH = "data/raw/tasks.json"
PROCESSED_PATH = "data/processed/"

os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load raw data
with open(RAW_PATH, "r") as f:
    data = json.load(f)

# Clean text
for d in data:
    d["task"] = clean_text(d["task"])
    d["input"] = clean_text(d["input"])

# Split into train/val/test
train, val, test = split_data(data)

# Save processed files
for split_name, split_data_set in zip(["train", "val", "test"], [train, val, test]):
    with open(os.path.join(PROCESSED_PATH, f"{split_name}.json"), "w") as f:
        json.dump(split_data_set, f, indent=2)

logger.info("Data preprocessing complete!")